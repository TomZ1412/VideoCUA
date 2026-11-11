# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge Qwen3.5 dense/MoE LoRA and multimodal trainable weights on CPU.

Adapted to this repository's checkpoint format from:
https://github.com/bytedance/video-SALMONN-2/blob/main/video_SALMONN2_pro/scripts/merge_base_model.py

--model_name_or_path is the full checkpoint used for LoRA training, while
--model_base supplies processor files if absent from that checkpoint.
The checkpoint tokenizer takes precedence so trained special tokens survive.

Example (replace the three input paths for a MoE checkpoint)::

    python scripts/merge_base_model.py \
        --model_name_or_path output/4B/checkpoint-7630 \
        --model_base output/4B/Qwen3.5-4B-Audio \
        --lora_ckpt output/4B/checkpoint-3000 --output_dir output/4B/merged
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor

from qwenvl.model.configuration_audio import Qwen3_5AudioConfig, Qwen3_5MoeAudioConfig
from qwenvl.model.modeling_qwen3_5_audio import model_class_for_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_base", help="Fallback tokenizer/processor directory")
    parser.add_argument("--lora_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_audio", action="store_true", help="For bases without an audio branch")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--video_longest_edge", type=int, default=235520000)
    return parser.parse_args()


def find_asset(directories, name):
    for directory in directories:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {name} in {directories}")


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    base = Path(args.model_name_or_path).resolve()
    adapter = Path(args.lora_ckpt).resolve()
    output = Path(args.output_dir).resolve()
    sources = [adapter, base, Path(args.model_base or base).resolve()]
    if output in sources or (output.exists() and any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    if args.video_longest_edge <= 0:
        raise ValueError("video_longest_edge must be positive")

    # Validate assets before loading the large model. Preserve Qwen3.5's image
    # settings and video minimum instead of importing Qwen3-VL patch-size limits.
    processors = {
        name: json.loads(find_asset(sources, name).read_text())
        for name in ("preprocessor_config.json", "video_preprocessor_config.json")
    }
    processors["video_preprocessor_config.json"]["size"]["longest_edge"] = args.video_longest_edge
    tokenizer = AutoTokenizer.from_pretrained(
        find_asset(sources, "tokenizer_config.json").parent, padding_side="right"
    )
    raw = json.loads((base / "config.json").read_text())
    config_classes = {"qwen3_5": Qwen3_5AudioConfig, "qwen3_5_moe": Qwen3_5MoeAudioConfig}
    if raw["model_type"] not in config_classes:
        raise ValueError(f"Unsupported model_type: {raw['model_type']}")
    if args.no_audio:
        if raw.get("audio_config"):
            raise ValueError("--no_audio cannot discard an existing audio branch")
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration
        config = AutoConfig.from_pretrained(base)
        model_class = (Qwen3_5MoeForConditionalGeneration if raw["model_type"] == "qwen3_5_moe"
                       else Qwen3_5ForConditionalGeneration)
    else:
        if not raw.get("audio_config"):
            raise ValueError("Audio merging requires an audio-initialized base checkpoint")
        config = config_classes[raw["model_type"]].from_pretrained(base)
        model_class = model_class_for_config(config)
        if tokenizer.convert_tokens_to_ids("<|audio_pad|>") != config.audio_token_id:
            raise ValueError("Tokenizer audio token does not match model config")

    weights = adapter / "trainable_weights.safetensors"
    if not args.no_audio and not weights.is_file():
        raise FileNotFoundError(f"Required audio/Q-Former trainable weights: {weights}")
    print(f"Loading {config.model_type} base: {base}", flush=True)
    model, info = model_class.from_pretrained(
        base, config=config, dtype=getattr(torch, args.dtype), device_map="cpu",
        attn_implementation="sdpa", output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete base checkpoint: {info}")

    if weights.is_file():
        model = get_peft_model(model, LoraConfig.from_pretrained(adapter))
        state = load_file(str(weights), device="cpu")
        required = {name for name, _ in model.named_parameters() if "lora_" in name}
        if required - state.keys():
            raise ValueError(f"Missing LoRA tensors: {sorted(required - state.keys())}")
        for name, tensor in state.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite trainable tensor: {name}")
        # Includes trained Q-Former, query tokens, audio projection and visual
        # merger; frozen Whisper encoder weights remain from the aligned base.
        _, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected trainable tensors: {unexpected}")
        audio_count = sum(".audio." in name for name in state)
        print(f"Loaded {len(state)} trainable tensors ({audio_count} audio)", flush=True)
        del state
    else:
        model = PeftModel.from_pretrained(model, adapter)

    print("Merging adapters", flush=True)
    model = model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("Unmerged LoRA parameters remain")
    model.eval()
    model.config.use_cache = True
    model.config.text_config.use_cache = True
    model.model.language_model.config.use_cache = True
    output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model: {output}", flush=True)
    model.save_pretrained(output, max_shard_size=args.max_shard_size, safe_serialization=True)
    tokenizer.save_pretrained(output)
    for name, value in processors.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    if not args.no_audio:
        # Keep Whisper metadata separate from the image preprocessor JSON.
        WhisperFeatureExtractor(
            feature_size=config.audio_config.num_mel_bins,
            sampling_rate=16000, hop_length=160, chunk_length=30,
        ).save_pretrained(output / "audio_processor")
    (output / "merge_info.json").write_text(json.dumps({
        "model_name_or_path": str(base), "lora_ckpt": str(adapter),
        "model_base": str(sources[-1]), "model_type": config.model_type,
        "dtype": args.dtype, "audio": not args.no_audio,
        "video_longest_edge": args.video_longest_edge,
    }, indent=2) + "\n")
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
