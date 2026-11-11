"""Qwen3.5 VS inference, using the local qwen3vl audio/data implementation."""
import copy
import os
import importlib
import json
import sys
from pathlib import Path

import torch


# Match the legacy VS loader: preprocessing assets live in a complete base
# directory, while model weights/config and optional LoRA come from checkpoints.
PROCESSOR_BASE_PATH = os.environ.get("VEX_PROCESSOR_BASE", "models/Qwen3.5-9B-Audio")


def _use_qwen35_package():
    # The migrated package still contains absolute `qwenvl.*` imports. Bind
    # those to qwen3vl only in this dedicated evaluation process.
    old = sys.modules.get("qwenvl")
    package = importlib.import_module("qwen35vl")
    if old is not None and list(old.__path__) != list(package.__path__):
        raise RuntimeError("Run Qwen3.5 VS in a separate process from legacy qwenvl")
    sys.modules["qwenvl"] = package


def load_model(lora_ckpt="No", model_name_or_path="Qwen/Qwen3.5-9B",
               max_pixels=2076192, max_video_pixels=128000, n_full_res=30,
               use_ttt=False, fixed_memory_size=0):
    if use_ttt or fixed_memory_size:
        raise ValueError("Legacy TTT is not supported by Qwen3.5 VS")
    _use_qwen35_package()
    from transformers import AutoConfig, AutoProcessor, WhisperFeatureExtractor
    from qwenvl.model.configuration_audio import Qwen3_5AudioConfig
    from qwenvl.model.modeling_qwen3_5_audio import model_class_for_config
    from qwenvl.train.argument import DataArguments

    raw = AutoConfig.from_pretrained(model_name_or_path)
    if raw.model_type != "qwen3_5":
        raise ValueError("--model-name must be a Qwen3.5-9B based VS checkpoint/base")
    config = Qwen3_5AudioConfig.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(PROCESSOR_BASE_PATH)
    if processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>") != config.audio_token_id:
        raise ValueError("Tokenizer audio token does not match VS config")
    model = model_class_for_config(config).from_pretrained(
        model_name_or_path, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if lora_ckpt and lora_ckpt.lower() not in ("no", "none"):
        from peft import PeftModel
        adapter = json.loads((Path(lora_ckpt) / "adapter_config.json").read_text())
        base = adapter.get("base_model_name_or_path", "").lower()
        if "qwen3-vl" in base or "qwen3_vl" in base:
            raise ValueError("Qwen3-VL LoRA cannot be loaded into Qwen3.5 VS")
        model = PeftModel.from_pretrained(model, lora_ckpt).merge_and_unload()
    model = model.to("cuda").eval()
    data_args = DataArguments(max_pixels=max_pixels, video_max_frame_pixels=max_video_pixels)
    data_args.video_processor = processor.video_processor
    data_args.image_processor = processor.image_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length, chunk_length=data_args.chunk_length,
    )
    # Match screenshot processing to the processor's native patch geometry.
    data_args.min_pixels = max(data_args.min_pixels, (processor.image_processor.patch_size * processor.image_processor.merge_size) ** 2)
    return model, processor, data_args


class _DatasetTokenizer:
    """The migrated dataset expects dictionary chat-template results."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        kwargs["return_dict"] = True
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class InferenceDataset:
    """Adapt legacy VS sources to Qwen3.5, retaining its video_bundle API."""
    def __init__(self, tokenizer, data_args):
        _use_qwen35_package()
        from qwenvl.data.data_qwen import LazySupervisedDataset
        self.dataset = LazySupervisedDataset(
            tokenizer=_DatasetTokenizer(tokenizer.tokenizer), data_args=data_args, inference=True, samples=[]
        )

    def _get_item(self, sources, video_bundle=None):
        sources = copy.deepcopy(sources)
        for source in sources:
            source["image"] = source.pop("images")
        data = self.dataset._get_item(sources, inference=True, no_meta=True)
        # Qwen3.5 computes multimodal RoPE from these token type ids.
        from qwenvl.data.data_qwen import multimodal_token_type_ids
        data["mm_token_type_ids"] = multimodal_token_type_ids(data["input_ids"], self.dataset.tokenizer)
        allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                   "pixel_values_videos", "video_grid_thw", "audio_feature", "mm_token_type_ids"}
        return {**{key: value for key, value in data.items() if key in allowed}, "video_bundle": None}
