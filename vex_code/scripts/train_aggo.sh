#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VEX_CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Paths vary across environments and must be supplied by the caller.
: "${DATASET_PATH:?Set DATASET_PATH to the AGGO training JSON file}"
: "${VALIDATION_DATA_PATH:?Set VALIDATION_DATA_PATH to the validation JSON file}"
: "${MODEL_PATH:?Set MODEL_PATH to the action-SFT model directory}"
: "${MODEL_BASE_PATH:?Set MODEL_BASE_PATH to the video-SALMONN 2+ base model}"
: "${ACTION_LORA_PATH:?Set ACTION_LORA_PATH to the action LoRA checkpoint}"
: "${GUIDANCE_LORA_PATH:?Set GUIDANCE_LORA_PATH to the guidance LoRA checkpoint}"

TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${VEX_CODE_DIR}/vsvex_train/cua_multirl_dualora/train.sh}"
RUN_NAME="${RUN_NAME:-vs2_agentnet_fromagentnet_GRPO_traindesc_realvideo_multiRL_bothLoRA_PPL_sample9_3}"

if [[ ! -f "${TRAIN_LAUNCHER}" ]]; then
    echo "Training launcher not found: ${TRAIN_LAUNCHER}" >&2
    exit 1
fi

# Three generation groups with beam size three in the trainer produce the nine
# candidate guidance samples reported for AGGO.
bash "${TRAIN_LAUNCHER}" \
    --interval 1.0 \
    --run_name "${RUN_NAME}" \
    --max_frames 512 \
    --save_steps 100 \
    --lr 2e-5 \
    --dataset "${DATASET_PATH}" \
    --validation_data "${VALIDATION_DATA_PATH}" \
    --model "${MODEL_PATH}" \
    --model_base "${MODEL_BASE_PATH}" \
    --epoch 2 \
    --model_type dense \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --do_validation \
    --total_timestamps 256 \
    --max_pixels_img 2076192 \
    --lora_ckpt "${ACTION_LORA_PATH}" \
    --lora_ckpt_2 "${GUIDANCE_LORA_PATH}" \
    --max_pixels 128000 \
    --train_type grpo \
    --grpo_num_sample 3 \
    --planner_type plan \
    --use_rloo \
    --action_samples 1 \
    --adapter_update both \
    --reward_func ppl
