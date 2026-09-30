#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(cd "${SCRIPT_DIR}/../vsvex_inference" && pwd)"

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running Gemini evaluation}"
: "${TASK_CONFIG_PATH:?Set TASK_CONFIG_PATH to the CUA-Gym 220-task JSON file}"
: "${VOLCENGINE_SUBNET_ID:?Set VOLCENGINE_SUBNET_ID for the OSWorld provider}"
: "${VOLCENGINE_ZONE_ID:?Set VOLCENGINE_ZONE_ID for the OSWorld provider}"
: "${VOLCENGINE_IMAGE_ID:?Set VOLCENGINE_IMAGE_ID for the OSWorld VM image}"
export GEMINI_API_KEY VOLCENGINE_SUBNET_ID VOLCENGINE_ZONE_ID VOLCENGINE_IMAGE_ID

export GEMINI_API_VERSION="${GEMINI_API_VERSION:-v1}"
RUN_ID="${RUN_ID:-gemini_35_cuagym_fixed}"
RESULT_DIR="${RESULT_DIR:-results}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_STEPS="${MAX_STEPS:-50}"

echo "${RESULT_DIR}/${RUN_ID}"
cd "${INFERENCE_DIR}"

python -m tests.test_gemini_local_cuagym \
    --mode formal \
    --task-config-path "${TASK_CONFIG_PATH}" \
    --num-workers "${NUM_WORKERS}" \
    --run-id "${RUN_ID}" \
    --result-dir "${RESULT_DIR}" \
    --max-steps "${MAX_STEPS}"
