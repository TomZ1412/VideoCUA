#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(cd "${SCRIPT_DIR}/../vsvex_inference" && pwd)"

: "${TASK_CONFIG_DIR:?Set TASK_CONFIG_DIR to the directory containing the sharded CUA-Gym task files}"
: "${MODEL_PATH:?Set MODEL_PATH to the video-SALMONN-VEX base model}"
: "${LORA_PATH:?Set LORA_PATH to the action LoRA checkpoint}"
: "${VOLCENGINE_SUBNET_ID:?Set VOLCENGINE_SUBNET_ID for the OSWorld provider}"
: "${VOLCENGINE_ZONE_ID:?Set VOLCENGINE_ZONE_ID for the OSWorld provider}"
: "${VOLCENGINE_IMAGE_ID:?Set VOLCENGINE_IMAGE_ID for the OSWorld VM image}"
export VOLCENGINE_SUBNET_ID VOLCENGINE_ZONE_ID VOLCENGINE_IMAGE_ID

RUN_ID="${RUN_ID:-vs2_agentnet_trainproj_cuagym220_fixed}"
RESULT_DIR="${RESULT_DIR:-results}"
GPU_COUNT="${GPU_COUNT:-8}"
SHARDS_PER_GPU="${SHARDS_PER_GPU:-3}"
NSAMPLES="${NSAMPLES:-3}"
ORACLE_TRAJ="${ORACLE_TRAJ:-none}"
GUIDE_TRAJ="${GUIDE_TRAJ:-none}"

echo "${RESULT_DIR}/${RUN_ID}"
cd "${INFERENCE_DIR}"

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    for ((offset = 0; offset < SHARDS_PER_GPU; offset++)); do
        idx=$((gpu * SHARDS_PER_GPU + offset))
        task_file="${TASK_CONFIG_DIR}/testset_full_${idx}.json"
        if [[ ! -f "${task_file}" ]]; then
            echo "Task shard not found: ${task_file}" >&2
            exit 1
        fi

        CUDA_VISIBLE_DEVICES="${gpu}" python -m tests.test_vss_local_cuagym \
            --run-id "${RUN_ID}" \
            --result-dir "${RESULT_DIR}" \
            --task-config-path "${task_file}" \
            --model-name "${MODEL_PATH}" \
            --lora-ckpt "${LORA_PATH}" \
            --nsamples "${NSAMPLES}" \
            --guide-traj "${GUIDE_TRAJ}" \
            --oracle-traj "${ORACLE_TRAJ}" &
    done
done

wait
