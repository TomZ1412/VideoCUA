"""Train Qwen3.5 Dense or MoE with audio using Transformers Trainer FSDP2."""

import os
import copy
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import transformers
from liger_kernel.transformers.monkey_patch import (
    apply_liger_kernel_to_qwen3_5,
    apply_liger_kernel_to_qwen3_5_moe,
)
from transformers import AutoProcessor, AutoTokenizer, WhisperFeatureExtractor

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.model.configuration_audio import Qwen3_5AudioConfig, Qwen3_5MoeAudioConfig
from qwenvl.model.modeling_qwen3_5_audio import model_class_for_config
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from qwenvl.train.prepare_model import prepare_model, prepare_generation_model
from qwenvl.train.trainer import FiniteLossTrainer


LANGUAGE_MODEL_LORA_TARGETS = (
    r".*language_model.*\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj))$"
)


def check_cua_adapter_architecture(path):
    config_file = Path(path) / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing LoRA adapter config: {config_file}")
    base = json.loads(config_file.read_text()).get("base_model_name_or_path") or ""
    if "qwen3-vl" in base.lower() or "qwen3_vl" in base.lower():
        raise ValueError(f"{path} is a Qwen3-VL adapter and cannot initialize Qwen3.5")
    base_config = Path(base) / "config.json"
    if base_config.is_file():
        model_type = json.loads(base_config.read_text()).get("model_type")
        if model_type not in ("qwen3_5", "qwen3_5_moe"):
            raise ValueError(f"{path} was trained on model_type={model_type}, not Qwen3.5")


def enable_liger_fused_ce_with_fsdp2():
    """Materialize the sharded LM head before Liger's Tensor-only fused loss."""
    from torch.distributed.tensor import DTensor
    from liger_kernel.transformers.model import qwen3_5, qwen3_5_moe

    for module in (qwen3_5, qwen3_5_moe):
        original_loss = module.LigerForCausalLMLoss

        def fsdp2_loss(*args, _original=original_loss, **kwargs):
            weight = kwargs.get("lm_head_weight")
            if isinstance(weight, DTensor):
                weight = weight.full_tensor()
            if weight is not None:
                # FSDP keeps optimizer parameters in fp32; the direct fused
                # projection bypasses nn.Linear's mixed-precision pre-hook.
                kwargs["lm_head_weight"] = weight.to(kwargs["hidden_states"].dtype)
            return _original(*args, **kwargs)

        module.LigerForCausalLMLoss = fsdp2_loss



def configure_trainable_parts(model, args):
    model.requires_grad_(False)
    if args.tune_mm_llm:
        if args.use_lora:
            raise ValueError("--tune_mm_llm and --use_lora cannot both be enabled")
        model.model.language_model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    if args.tune_mm_vision:
        model.model.visual.requires_grad_(True)
    if args.tune_mm_mlp:
        model.model.visual.merger.requires_grad_(True)
    if args.tune_mm_audio:
        model.model.audio.requires_grad_(True)
    if args.tune_mm_qformer:
        model.model.audio.qformer.requires_grad_(True)
        model.model.audio.q_tokens.requires_grad_(True)
        model.model.audio.audio_proj.requires_grad_(True)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.train_type == "grpo" and training_args.grpo_response_batch_size < 1:
        raise ValueError("grpo_response_batch_size must be positive")
    if training_args.train_type not in ("sft", "grpo", "cua_grpo"):
        raise ValueError("train_type must be sft, grpo, or cua_grpo")
    is_cua = training_args.train_type == "cua_grpo"
    if is_cua and not model_args.use_lora:
        raise ValueError("CUA GRPO requires --use_lora")
    if training_args.do_validation and not training_args.validation_data:
        raise ValueError("--do_validation requires --validation_data")
    if is_cua and any((model_args.tune_mm_llm, model_args.tune_mm_mlp,
                       model_args.tune_mm_vision, model_args.tune_mm_audio,
                       model_args.tune_mm_qformer)):
        raise ValueError("CUA GRPO currently trains only the two language adapters")
    training_args.remove_unused_columns = False
    if training_args.deepspeed is not None:
        raise ValueError("DeepSpeed is no longer supported; launch with scripts/train.sh for FSDP2")
    if not training_args.fsdp:
        raise ValueError("FSDP2 must be enabled with --fsdp true")
    if training_args.gradient_checkpointing and training_args.fsdp_config.get("activation_checkpointing"):
        raise ValueError("Choose either gradient checkpointing or FSDP activation checkpointing")

    if training_args.resume_from_checkpoint:
        raise ValueError("resume_from_checkpoint is no longer supported; use --lora_ckpt to merge a new base")
    config_path = str(prepare_model(
        model_args.model_name_or_path, model_args.model_base,
        None if is_cua else model_args.lora_ckpt, training_args.output_dir,
    ))
    merged_base = (Path(config_path) / "train_merge.json").is_file()
    # Initialize the distributed process group before querying its global rank.
    _ = training_args.device
    base_upload = None
    if merged_base and training_args.hdfs_output_dir and dist.get_rank() == 0:
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        base_upload = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/upload_checkpoint.py"),
             config_path, training_args.hdfs_output_dir, "--save_total_limit", "0"],
            env=env,
        )
    raw_config = transformers.AutoConfig.from_pretrained(config_path)
    if raw_config.model_type not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError(f"Unsupported model_type: {raw_config.model_type}")
    config_class = Qwen3_5MoeAudioConfig if raw_config.model_type == "qwen3_5_moe" else Qwen3_5AudioConfig
    config = config_class.from_pretrained(config_path)
    model_max_length = 262_144
    if config.text_config.max_position_embeddings != model_max_length:
        raise ValueError(
            f"Expected Qwen3.5 native context length {model_max_length}, "
            f"got {config.text_config.max_position_embeddings}"
        )
    model_class = model_class_for_config(config)
    if config.model_type == "qwen3_5_moe":
        apply_liger_kernel_to_qwen3_5_moe(fused_linear_cross_entropy=True, swiglu=True)
    else:
        apply_liger_kernel_to_qwen3_5(fused_linear_cross_entropy=True)
    enable_liger_fused_ce_with_fsdp2()

    base_path = config_path if merged_base else (model_args.model_base or config_path)
    processor = AutoProcessor.from_pretrained(base_path)
    data_args.video_processor = processor.video_processor
    data_args.image_processor = processor.image_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side="right")
    tokenizer.model_max_length = model_max_length
    if tokenizer.convert_tokens_to_ids("<|audio_pad|>") != config.audio_token_id:
        raise ValueError("Tokenizer audio token does not match model config")
    if is_cua:
        from qwenvl.data.cua_data import CUADualLoRADataset, collate_cua
        from qwenvl.train.cua_trainer import CUADualLoRATrainer

        data_module = dict(
            train_dataset=CUADualLoRADataset(tokenizer, data_args, data_args.dataset_use),
            eval_dataset=(CUADualLoRADataset(tokenizer, data_args, training_args.validation_data)
                          if training_args.do_validation else None),
            data_collator=collate_cua,
        )
        trainer_class = CUADualLoRATrainer
    elif training_args.train_type == "grpo":
        from qwenvl.data.grpo_data import GRPODataset, collate_prompt_indices
        from qwenvl.train.grpo_trainer import GRPOTrainer
        data_module = dict(train_dataset=GRPODataset(tokenizer, data_args),
                           data_collator=collate_prompt_indices)
        data_module["generation_model_path"] = str(prepare_generation_model(config_path, base_path, training_args.output_dir))
        trainer_class = GRPOTrainer
    else:
        data_module = make_supervised_data_module(tokenizer, data_args)
        if training_args.do_validation:
            from qwenvl.data.data_qwen import LazySupervisedDataset
            eval_data_args = copy.copy(data_args)
            eval_data_args.dataset_use = training_args.validation_data
            eval_data_args.length_cache = None
            data_module["eval_dataset"] = LazySupervisedDataset(tokenizer, eval_data_args)
            if len(data_module["eval_dataset"]) == 0:
                raise ValueError("Validation dataset is empty")
        trainer_class = FiniteLossTrainer

    model = model_class.from_pretrained(
        config_path,
        config=config,
        dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=model_args.attn_implementation,
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.model.language_model.config.use_cache = False
    if training_args.train_type == "grpo" and training_args.grpo_response_batch_size > 1:
        from qwenvl.model.grpo_shared_prefix import enable_shared_prefix
        enable_shared_prefix(model)
    configure_trainable_parts(model, model_args)
    if is_cua:
        from peft import LoraConfig, PeftModel, get_peft_model
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        if bool(model_args.lora_ckpt) != bool(model_args.lora_ckpt_2):
            raise ValueError("Supply both --lora_ckpt and --lora_ckpt_2, or neither")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.0,
            target_modules=LANGUAGE_MODEL_LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        if model_args.lora_ckpt:
            check_cua_adapter_architecture(model_args.lora_ckpt)
            check_cua_adapter_architecture(model_args.lora_ckpt_2)
            model = PeftModel.from_pretrained(
                model, model_args.lora_ckpt, adapter_name="adapter_1", is_trainable=True)
            model.load_adapter(model_args.lora_ckpt, adapter_name="adapter_1_ref",
                               is_trainable=False)
            model.load_adapter(model_args.lora_ckpt_2, adapter_name="adapter_2",
                               is_trainable=True)
            model.load_adapter(model_args.lora_ckpt_2, adapter_name="adapter_2_ref",
                               is_trainable=False)
        else:
            model = get_peft_model(model, lora_config, adapter_name="adapter_1")
            for source, reference in (("adapter_1", "adapter_1_ref"),
                                      ("adapter_2", "adapter_2_ref")):
                if source == "adapter_2":
                    model.add_adapter(source, copy.deepcopy(lora_config))
                model.add_adapter(reference, copy.deepcopy(lora_config))
                state = get_peft_model_state_dict(model, adapter_name=source)
                set_peft_model_state_dict(model, state, adapter_name=reference)
        model.set_adapter("adapter_1")
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(
                    ".adapter_1." in name or ".adapter_2." in name)
        for adapter in ("adapter_1", "adapter_2"):
            if not any("lora_" in name and f".{adapter}." in name and parameter.requires_grad
                       for name, parameter in model.named_parameters()):
                raise RuntimeError(f"{adapter} has no trainable Qwen3.5 LoRA weights")
    elif model_args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=LANGUAGE_MODEL_LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(
            model,
            lora_config,
        )
        if not any("lora_" in name and param.requires_grad for name, param in model.named_parameters()):
            raise RuntimeError("LoRA adapters are frozen; cannot continue SFT")
        core = model.base_model.model
        if model_args.tune_mm_audio:
            core.model.audio.requires_grad_(True)
        elif model_args.tune_mm_qformer:
            core.model.audio.qformer.requires_grad_(True)
            core.model.audio.q_tokens.requires_grad_(True)
            core.model.audio.audio_proj.requires_grad_(True)
        if model_args.tune_mm_mlp:
            core.model.visual.merger.requires_grad_(True)
        if model_args.tune_mm_vision:
            core.model.visual.requires_grad_(True)

    if dist.get_rank() == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable paramerters: {trainable}")
        for k, v in model.named_parameters():
            if v.requires_grad:
                print(k, v.shape)
    trainer = trainer_class(model=model, processing_class=tokenizer,
                                   args=training_args, **data_module)
    if base_upload is not None:
        trainer._hdfs_uploads = [base_upload]
    trainer.train()
    if training_args.train_type == "sft" and training_args.do_validation:
        trainer.evaluate()
    trainer.wait_for_uploads()
    trainer.save_state()
    if training_args.save_strategy != "no":
        trainer.save_model()


if __name__ == "__main__":
    train()
