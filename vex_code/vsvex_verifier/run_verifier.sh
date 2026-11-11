#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to your Qwen3.5 checkpoint}"
: "${MODEL_BASE:?Set MODEL_BASE to your processor/base checkpoint}"
: "${VERIFIER_DATA:?Set VERIFIER_DATA to your verifier input JSON}"

extra_args=()
if [[ "${NO_AUDIO:-false}" == true ]]; then
  extra_args+=(--no_audio)
fi

cd "$SCRIPT_DIR"
bash test.sh \
  --num_gpus "${NUM_GPUS:-8}" \
  --interval "${INTERVAL:-1}" \
  --run_name "${RUN_NAME:-qwen35_verifier}" \
  --dataset "$VERIFIER_DATA" \
  --max_frames "${MAX_FRAMES:-768}" \
  --model "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --model_type "${MODEL_TYPE:-dense}" \
  --max_pixels "${MAX_PIXELS:-128000}" \
  --lora_ckpt "${LORA_CKPT:-No}" \
  --expectation "${EXPECTATION:-true}" \
  "${extra_args[@]}" "$@"
