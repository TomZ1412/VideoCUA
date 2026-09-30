from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
import torch
import shutil
import os

model_source = os.environ.get("MODEL_SOURCE", "/path/to/Qwen3-VL-2B-Instruct")
model_output = os.environ.get("MODEL_OUTPUT", "/path/to/Qwen3-VL-2B-Instruct-Audio-MoreToken")
whisper_model_path = os.environ.get("WHISPER_MODEL_PATH", "/path/to/whisper-large-v3")

# def apply_fused_kernel_to_moe():
#     print("Applying fused kernel to MoE")
#     from qwenvl.model.qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock
#     from qwenvl.model import modeling_qwen3_vl_moe

#     modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = Qwen3MoeFusedSparseMoeBlock

# apply_fused_kernel_to_moe()


tokenizer = AutoTokenizer.from_pretrained(
    model_source,
    padding_side="right",
    use_fast=False,
)

tokenizer.add_tokens(["<|audio_pad|>"])
# 
tokenizer.save_pretrained(model_output)

attn_implementation="flash_attention_2"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_source,
    attn_implementation=attn_implementation,
    torch_dtype=(torch.bfloat16),
)

model_id = whisper_model_path

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)

for k, v in model.model.audio.named_parameters():
    if k in whisper_model.model.encoder.state_dict() and v.shape == whisper_model.model.encoder.state_dict()[k].shape:
        v.data = whisper_model.model.encoder.state_dict()[k].data
    else:
        print(k)

model.model.audio.q_tokens.data.normal_(mean=0.0, std=0.02)

print("-------")

model.save_pretrained(model_output)

shutil.copy(
    os.path.join(model_source, "chat_template.json"),
    os.path.join(model_output, "chat_template.json")
)

shutil.copy(
    os.path.join(model_source, "preprocessor_config.json"),
    os.path.join(model_output, "preprocessor_config.json")
)

shutil.copy(
    os.path.join(model_source, "video_preprocessor_config.json"),
    os.path.join(model_output, "video_preprocessor_config.json")
)

for k, v in model.named_parameters():
    if (v>100).any():
        print(k)

# breakpoint()
