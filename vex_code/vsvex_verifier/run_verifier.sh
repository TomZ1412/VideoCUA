#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET=""
MODEL=""
MODEL_BASE=""
LORA_CKPT="No"
RUN_NAME="vex_verifier"
GPU_COUNT="${GPU_COUNT:-1}"
INTERVAL=1
MAX_FRAMES=768
MAX_PIXELS=128000
MAX_PIXELS_IMG=2076192
TOTAL_TIMESTAMPS=256
MODEL_TYPE=dense
EXPECTATION=False
OUTPUT_DIR="${OUTPUT_DIR:-output/test}"
NUM_WORKERS="${NUM_WORKERS:-8}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model_base) MODEL_BASE="$2"; shift 2 ;;
        --lora_ckpt) LORA_CKPT="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --gpu_count) GPU_COUNT="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --max_frames) MAX_FRAMES="$2"; shift 2 ;;
        --max_pixels) MAX_PIXELS="$2"; shift 2 ;;
        --max_pixels_img) MAX_PIXELS_IMG="$2"; shift 2 ;;
        --total_timestamps) TOTAL_TIMESTAMPS="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --expectation) EXPECTATION=True; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

: "${DATASET:?Pass --dataset with the verifier input JSON}"
: "${MODEL:?Pass --model with the verifier model path}"
: "${MODEL_BASE:?Pass --model_base with the video-SALMONN 2+ base model path}"

if [[ ! -f "${DATASET}" ]]; then
    echo "Dataset not found: ${DATASET}" >&2
    exit 1
fi
if [[ ! -f "scripts/zero2.json" ]]; then
    echo "DeepSpeed config not found: scripts/zero2.json" >&2
    exit 1
fi

SHARD_DIR="dataset/${RUN_NAME}"
mkdir -p "${SHARD_DIR}" "${OUTPUT_DIR}/${RUN_NAME}"
python scripts/split_data.py "${DATASET}" "${GPU_COUNT}" "${SHARD_DIR}"

for ((rank = 0; rank < GPU_COUNT; rank++)); do
    CUDA_VISIBLE_DEVICES="${rank}" torchrun \
        --nproc_per_node=1 \
        --master_port="$((12345 + rank))" \
        qwenvl/train/train_qwen.py \
        --model_base "${MODEL_BASE}" \
        --run_test True \
        --pred_rank "${rank}" \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "${MODEL}" \
        --dataset_use "${SHARD_DIR}/${rank}.json" \
        --tune_mm_vision False \
        --tune_mm_mlp False \
        --tune_mm_llm False \
        --bf16 \
        --output_dir "${OUTPUT_DIR}" \
        --per_device_train_batch_size 1 \
        --max_pixels "${MAX_PIXELS_IMG}" \
        --min_pixels 784 \
        --video_max_frame_pixels "${MAX_PIXELS}" \
        --video_min_frame_pixels 784 \
        --eval_strategy no \
        --save_strategy no \
        --model_max_length 1310720 \
        --dataloader_num_workers "${NUM_WORKERS}" \
        --run_name "${RUN_NAME}" \
        --report_to none \
        --video_min_frames 4 \
        --video_max_frames "${MAX_FRAMES}" \
        --base_interval "${INTERVAL}" \
        --lora_ckpt "${LORA_CKPT}" \
        --total_timestamps "${TOTAL_TIMESTAMPS}" \
        --expectation "${EXPECTATION}" \
        --model_type "${MODEL_TYPE}" &
done

wait
python scripts/merge_json.py "${OUTPUT_DIR}/${RUN_NAME}" "${GPU_COUNT}"
