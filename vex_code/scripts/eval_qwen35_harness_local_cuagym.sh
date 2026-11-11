#!/usr/bin/env bash
set -euo pipefail

export VOLCENGINE_SUBNET_ID="${VOLCENGINE_SUBNET_ID:?Set VOLCENGINE_SUBNET_ID}"
export VOLCENGINE_ZONE_ID="${VOLCENGINE_ZONE_ID:?Set VOLCENGINE_ZONE_ID}"
export VOLCENGINE_ACCESS_KEY_ID="${VOLCENGINE_ACCESS_KEY_ID:?Set VOLCENGINE_ACCESS_KEY_ID}"
export VOLCENGINE_SECRET_ACCESS_KEY="${VOLCENGINE_SECRET_ACCESS_KEY:?Set VOLCENGINE_SECRET_ACCESS_KEY}"
export VOLCENGINE_IMAGE_ID="${VOLCENGINE_IMAGE_ID:?Set VOLCENGINE_IMAGE_ID}"
export VEX_PROCESSOR_BASE="${VEX_PROCESSOR_BASE:?Set VEX_PROCESSOR_BASE}"

output_dir="${RUN_ID:?Set RUN_ID}"
base_task_config_dir="${TASK_CONFIG_DIR:?Set TASK_CONFIG_DIR}"
base_model="${MODEL_PATH:?Set MODEL_PATH}"
base_lora="${LORA_CKPT:-No}"
guide_traj="${GUIDE_TRAJ:-none}"
oracle_traj="${ORACLE_TRAJ:-none}"
num_gpus="${NUM_GPUS:-8}"
tasks_per_gpu="${TASKS_PER_GPU:-3}"
max_steps="${MAX_STEPS:-50}"
n_history="${N_HISTORY:-5}"
nsamples="${NSAMPLES:-1}"
max_tokens="${MAX_TOKENS:-4096}"
result_dir="${RESULT_DIR:-results}"

echo "$output_dir"

for ((gpu = 0; gpu < num_gpus; gpu++)); do
  export CUDA_VISIBLE_DEVICES="$gpu"
  for ((offset = 0; offset < tasks_per_gpu; offset++)); do
    idx=$((gpu * tasks_per_gpu + offset))
    python -m tests.test_qwen35_harness_local \
      --run-id "$output_dir" \
      --result-dir "$result_dir" \
      --task-config-path "${base_task_config_dir}/testset_full_${idx}.json" \
      --model-name "$base_model" \
      --lora-ckpt "$base_lora" \
      --guide-traj "$guide_traj" \
      --oracle-traj "$oracle_traj" \
      --benchmark cuagym \
      --max-steps "$max_steps" \
      --n-history "$n_history" \
      --nsamples "$nsamples" \
      --max-tokens "$max_tokens" &
  done
done

wait
