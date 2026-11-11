"""Load VS2 audio-visual checkpoints with a Qwen3.5 Dense/MoE backbone."""
import json
from pathlib import Path


def read_records(path):
    text = Path(path).read_text()
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ValueError('Dataset must be a JSON array or JSONL')
    return records



def merge_adapter_in_memory(model, adapter_path):
    """Match the harness, restoring supplemental trainable weights if present."""
    import torch
    from peft import PeftModel
    adapter_path = Path(adapter_path)
    adapter = json.loads((adapter_path / 'adapter_config.json').read_text())
    base = (adapter.get('base_model_name_or_path') or '').lower()
    if 'qwen3-vl' in base or 'qwen3_vl' in base:
        raise ValueError('Qwen3-VL LoRA cannot initialize Qwen3.5')
    model = PeftModel.from_pretrained(model, str(adapter_path))
    weights = adapter_path / 'trainable_weights.safetensors'
    if weights.is_file():
        from safetensors.torch import load_file
        state = load_file(str(weights), device='cpu')
        required = {name for name, _ in model.named_parameters() if 'lora_' in name}
        if required - state.keys():
            raise ValueError(f'Missing LoRA tensors in {weights}: {sorted(required - state.keys())}')
        for name, tensor in state.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f'Non-finite trainable tensor: {name}')
        _, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f'Unexpected trainable tensors: {unexpected}')
        print(f'Restored {len(state)} supplemental trainable tensors from {weights}', flush=True)
    return model.merge_and_unload(safe_merge=True)


def load_model(model_path, processor_path=None, device='cuda', dtype='bfloat16',
               attn_implementation='sdpa', lora=None, merge_dir=None):
    import torch
    from transformers import AutoConfig, AutoProcessor, WhisperFeatureExtractor
    from qwenvl.train.argument import DataArguments

    if lora and lora.lower() in ("no", "none"):
        lora = None
    if lora and merge_dir is not None:
        from qwenvl.train.prepare_model import prepare_model
        model_path = str(prepare_model(model_path, processor_path, lora, merge_dir))
        processor_path = model_path
        lora = None
    raw = AutoConfig.from_pretrained(model_path)
    if raw.model_type not in ('qwen3_5', 'qwen3_5_moe'):
        raise ValueError(f'Expected a Qwen3.5 checkpoint, got {raw.model_type}')
    has_audio = bool(getattr(raw, 'audio_config', None))
    if not has_audio:
        raise ValueError('Verifier requires a VS2 Qwen3.5 audio-visual checkpoint, not native Qwen3.5')
    from qwenvl.model.configuration_audio import Qwen3_5AudioConfig, Qwen3_5MoeAudioConfig
    from qwenvl.model.modeling_qwen3_5_audio import model_class_for_config
    config_cls = Qwen3_5MoeAudioConfig if raw.model_type == 'qwen3_5_moe' else Qwen3_5AudioConfig
    config = config_cls.from_pretrained(model_path)
    model_cls = model_class_for_config(config)
    processor = AutoProcessor.from_pretrained(processor_path or model_path)
    if has_audio and processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>') != config.audio_token_id:
        raise ValueError('Processor audio token does not match model config')
    model, info = model_cls.from_pretrained(
        model_path, config=config, dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Incomplete checkpoint loading: {info}')
    if lora:
        model = merge_adapter_in_memory(model, lora)
    model = model.to(device).eval()
    data_args = DataArguments()
    data_args.video_processor = processor.video_processor
    data_args.image_processor = processor.image_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length, chunk_length=data_args.chunk_length)
    geometry = (processor.image_processor.patch_size * processor.image_processor.merge_size) ** 2
    data_args.min_pixels = max(data_args.min_pixels, geometry)
    data_args.video_min_frame_pixels = max(data_args.video_min_frame_pixels, geometry)
    return model, processor, data_args, has_audio
