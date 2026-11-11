import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_base: Optional[str] = field(default=None)
    attn_implementation: str = field(default="sdpa")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_audio: bool = field(default=False)
    tune_mm_qformer: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    lora_bias: str = field(default="none")
    lora_ckpt: Optional[str] = field(default=None)
    lora_ckpt_2: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    thinking: Optional[bool] = field(default=None)
    dataset_use: str = field(default="")
    length_cache: Optional[str] = field(default=None)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: float = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    feature_size: int = field(default=128)
    chunk_length: int = field(default=30)
    hop_length: int = field(default=160)
    sampling_rate: int = field(default=16000)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    train_type: str = field(default="sft")
    planner_type: str = field(default="plan")
    reward_func: str = field(default="ppl")
    adapter_update: str = field(default="both")
    grpo_num_sample: int = field(default=3)
    action_samples: int = field(default=1)
    use_rloo: bool = field(default=False)
    cua_kl_beta: float = field(default=0.05)
    cua_max_plan_tokens: int = field(default=1024)
    cua_max_action_tokens: int = field(default=1024)
    validation_data: Optional[str] = field(default=None)
    do_validation: bool = field(default=False)
    num_generations: int = field(default=8)
    grpo_response_batch_size: int = field(default=1, metadata={
        "help": "Responses sharing one differentiable prompt forward/backward (positive integer)."})
    max_output_token: int = field(default=16384)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    top_k: int = field(default=20)
    repetition_penalty: float = field(default=1.0)
    reward_weights: str = field(default="0.05,1,1")
    epsilon: float = field(default=0.2)
    loss_type: str = field(default="dapo", metadata={"choices": ["dapo", "grpo"]})
    importance_sampling_level: str = field(default="token", metadata={"choices": ["token", "sequence"]})
    dynamic_sampling: bool = field(default=False)
    dynamic_sampling_oversample_factor: float = field(default=1.2)
    dynamic_sampling_initial_accept_rate: float = field(default=0.5)
    dynamic_sampling_max_factor: float = field(default=8.0)
    vllm_importance_sampling_cap: float = field(default=2.0)
    vllm_tensor_parallel_size: int = field(default=1)
    vllm_gpu_memory_utilization: float = field(default=0.45)
    vllm_max_model_len: int = field(default=120000)
    vllm_max_num_batched_tokens: int = field(default=8192)
    grpo_offload_model: bool = field(default=True)
    sync_vllm: bool = field(default=True)
    vllm_enforce_eager: bool = field(default=False)

    cache_dir: Optional[str] = field(default=None)
    hdfs_output_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    no_audio: bool = field(default=False)
    use_modality_sampler: bool = field(default=False)
    length_grouping_window: int = field(default=64)
