# Qwen3.5 CUA dual-LoRA GRPO

This branch starts from the native Qwen3.5 audio model and FSDP2 trainer. The
CUA path uses a video plus first screenshot to generate a plan with
`adapter_2`, then uses screenshot history to score or sample the final action
with `adapter_1`. In PPL mode the plan reward is
`NLL(action without plan) - NLL(action with plan)`. W&B logs both NLLs and
the gap. As in the original CUA branch, `grpo_num_sample=3` produces nine
plan candidates (three sampled completions per group), PPL covers all unmasked
historical assistant actions, and policy gradients use the final action. PPL
action updates do not have an extra reference KL; plan updates do. Validation
generates one plan and one action per example with the original seven metrics.

The root `train.sh` is the primary launcher, as in `cua_multirl_dualora`;
`scripts/train.sh` delegates to it. The model-independent legacy evaluation
and dataset utilities retain their original paths. Qwen3-VL model modules,
TTT kernels, DeepSpeed/FSDP1 configs, and their old-model test launchers are
not copied because they do not run against the Qwen3.5 backbone.

Launch from this checkout with a **full Qwen3.5 audio checkpoint**. The two
legacy Qwen3-VL LoRA checkpoints cannot be loaded into Qwen3.5; omit both
checkpoint flags to initialize two new Qwen3.5 adapters, or provide two
Qwen3.5-compatible adapter directories.

```bash
bash ./train.sh \
  --interval 1.0 \
  --run_name qwen35_cua_dualora_ppl \
  --max_frames 512 \
  --max_pixels 128000 \
  --max_pixels_img 2076192 \
  --use_lora --lora_r 128 --lora_alpha 256 \
  --save_steps 100 --lr 2e-5 --epoch 1 \
  --dataset /path/to/agentnet_train_filter_basetrainproj_hard_more_with_video_realonly.json \
  --validation_data /path/to/agentnet_valid_with_real_video.json \
  --model /path/to/checkpoint-7630 \
  --model_base /path/to/checkpoint \
  --hdfs_output_dir /path/to/checkpoint \
  --train_type cua_grpo --planner_type plan --reward_func ppl \
  --grpo_num_sample 3 --action_samples 1 --adapter_update both \
  --use_rloo --do_validation
```

The saved checkpoint contains `adapter_1/` and `adapter_2/`. To initialize
from a Qwen3.5 dual-adapter checkpoint, pass
`--lora_ckpt PATH/adapter_1 --lora_ckpt_2 PATH/adapter_2`.

This path requires per-device batch size 1 and accumulation 1. It uses native
model generation on every FSDP rank, so a small cluster smoke test should
precede a full run. The single-response vLLM GRPO path remains available as
`--train_type grpo`.
