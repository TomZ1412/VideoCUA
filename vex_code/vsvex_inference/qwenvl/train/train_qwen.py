# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import random
import time

from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.model.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.data_qwen import make_supervised_data_module, LazySupervisedDataset
from qwenvl.data.processing_qwen3_vl import Qwen3VLProcessor
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from tqdm import tqdm
import torch.distributed as dist

try:
    from cruise.utilities.distributed import DIST_ENV
    from cruise.utilities.hdfs_io import hcopy, hmkdir
except:
    pass

# import qwenvl.train.deepspeed_patch

HDFS_BASE_PATH = ""

local_rank = None

def collate_fn(batch):
    return batch[0]

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def apply_fused_kernel_to_moe():
    print("Applying fused kernel to MoE")
    from qwenvl.model.qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock
    from qwenvl.model import modeling_qwen3_vl_moe

    modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = Qwen3MoeFusedSparseMoeBlock

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    print("Applying Liger kernels to Qwen3-VL model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model import modeling_qwen3_vl_moe
    from qwenvl.model import modeling_qwen3_vl

    if rms_norm:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm = LigerRMSNorm
        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextMLP = LigerSwiGLUMLP
        modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        model.model.visual.requires_grad_(True)
    else:
        model.model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.model.visual.merger.requires_grad_(True)
    else:
        model.model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.model.audio.requires_grad_(True)
    else:
        model.model.audio.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.model.audio.qformer.requires_grad_(True)
        model.model.audio.q_tokens.requires_grad_(True)
        model.model.audio.audio_proj.requires_grad_(True)
    else:
        model.model.audio.qformer.requires_grad_(False)
        model.model.audio.q_tokens.requires_grad_(False)
        model.model.audio.audio_proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.language_model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.language_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)


def load_model(
    lora_ckpt="No",
    model_name_or_path = "path_to_model",
    attn_implementation="flash_attention_2",
    max_pixels=2076192,
    max_video_pixels=128000,
    n_full_res=30,
    use_ttt=False,
    fixed_memory_size=0,
):
    apply_liger_kernel_to_qwen2_5_vl()

    model_base = "Qwen3-VL-8B-Instruct-Audio"
    video_processor = Qwen3VLProcessor.from_pretrained(
        model_base,
    ).video_processor
    image_processor = Qwen3VLProcessor.from_pretrained(
        model_base,
    ).image_processor
    audio_processor = WhisperFeatureExtractor(
        feature_size=128, 
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_base,
        cache_dir=None,
        padding_side="right",
        use_fast=False,
    )

    # os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

    if torch.cuda.current_device() == 0:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )

        # Stream
        if use_ttt:
            model.model.fixed_memory_size = fixed_memory_size
            model.model.fixed_memory_size_audio = 4096
            model.model.stepsize = 4096
            model.model.ttt_type = "ttt_simsample"
            model.model.init_ttt_layers(
                num_heads=8,
                ttt_hidden_size=1,
                CG_max_iter=0,
                slot_type="lite",
                ema_factor=0.1,
                lag_distances="1",
            )

    if lora_ckpt != "No":
        if torch.cuda.current_device() == 0:
            from peft import PeftModel
            audio_layers = model.model.audio.layers
            del model.model.audio.layers
            model = PeftModel.from_pretrained(model, lora_ckpt)
            model = model.to(torch.bfloat16)
            model.model.model.audio.layers = audio_layers
            model = model.merge_and_unload()
    else:
        model = model.to(torch.bfloat16)

    model.cuda()
    data_args = DataArguments()
    data_args.n_full_res = n_full_res
    data_args.max_pixels = max_pixels
    data_args.video_max_frame_pixels = max_video_pixels
    data_args.video_processor = video_processor
    data_args.image_processor = image_processor
    data_args.audio_processor = audio_processor
    return model, tokenizer, data_args


class DataArguments:
    dataset_use: str = "path_to_dataset"
    video_max_frames: int = 512
    video_min_frames: int = 2
    base_interval: float = 1
    max_pixels: int = 2076192  # 922752 for 720p
    min_pixels: int = 28 * 28 *  16
    video_max_frame_pixels: int = 128000
    video_min_frame_pixels: int = 768
    run_test: bool = True
    do_sample: bool = False
    num_sample: int = 1
    train_type: str = "sft"
    validation_data: str = ""
    total_timestamps: int = 256
    model_type: str = "qwen3vl"
    n_full_res: int = 30


if __name__ == "__main__":
    model, tokenizer, data_args = load_model(attn_implementation="flash_attention_3")
    inference(model, tokenizer, data_args)