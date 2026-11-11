#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to your model checkpoint}"
: "${MODEL_BASE:?Set MODEL_BASE to your processor/base checkpoint}"
: "${TRAIN_DATA:?Set TRAIN_DATA to your training JSON}"
: "${VALIDATION_DATA:?Set VALIDATION_DATA to your validation JSON}"
: "${LORA_CKPT:?Set LORA_CKPT to your initialization adapter}"

cd "$SCRIPT_DIR/qwen35_cua_multirl_dualora"
bash train.sh \
  --interval "${INTERVAL:-1.0}" \
  --run_name "${RUN_NAME:-sft_desc}" \
  --max_frames "${MAX_FRAMES:-512}" \
  --use_lora \
  --save_steps "${SAVE_STEPS:-2000}" \
  --lr "${LEARNING_RATE:-2e-5}" \
  --dataset "$TRAIN_DATA" \
  --validation_data "$VALIDATION_DATA" \
  --model "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --lora_ckpt "$LORA_CKPT" \
  --epoch "${EPOCHS:-3}" \
  --model_type "${MODEL_TYPE:-dense}" \
  --lora_r "${LORA_R:-128}" \
  --lora_alpha "${LORA_ALPHA:-256}" \
  --do_validation \
  --total_timestamps "${TOTAL_TIMESTAMPS:-256}" \
  --max_pixels_img "${MAX_PIXELS_IMG:-2076192}" \
  --max_pixels "${MAX_PIXELS:-128000}" \
  --train_type sft \
  --no_hdfs_upload \
  --report_to "${REPORT_TO:-none}" \
  "$@"
