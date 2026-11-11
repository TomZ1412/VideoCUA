#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export NCCL_DEBUG="${QWENVL_NCCL_DEBUG:-WARN}"

TRAIN_TYPE="sft"
GRPO_ARGS=()
MODEL=""
MODEL_BASE=""
MODEL_TYPE=""
TOTAL_TIMESTAMPS=""
LORA_CKPT=""
LORA_CKPT_2=""
DATASET=""
VALIDATION_DATA=""
RUN_NAME="qwen35_audio"
REPORT_TO="wandb"
LR="2e-5"
BS="1"
ACCUM_STEPS="1"
EPOCH="1"
MAX_STEPS="-1"
SAVE_STEPS="1000"
MAX_PIXELS="176400"
MAX_PIXELS_IMG="13107200"
MIN_PIXELS="784"
MIN_PIXELS_IMG="784"
MIN_FRAMES="64"
MAX_FRAMES="128"
INTERVAL="0.2"
NUM_WORKER="8"
PREFETCH_FACTOR="2"
WARMUP_RATIO="0.03"
LORA_R="128"
LORA_ALPHA="256"
LORA_DROPOUT="0.05"
SHARD_MODEL=false
NO_CHECKPOINT=false
ATTN_IMPLEMENTATION="flash_attention_3"
HDFS_UPLOAD=false
HDFS_OUTPUT_DIR=""
TRAIN_LLM=false
TRAIN_PROJ=false
TRAIN_ENC=false
TRAIN_AUDIO=false
TRAIN_QFORMER=false
USE_LORA=false
DO_VALIDATION=false
GRPO_NUM_SAMPLE="0"
PLANNER_TYPE="none"
ACTION_SAMPLES="1"
ADAPTER_UPDATE="both"
REWARD_FUNC="sample"
CUA_KL_BETA="0.05"
CUA_MAX_PLAN_TOKENS="1024"
CUA_MAX_ACTION_TOKENS="1024"
USE_RLOO=false
LENGTH_CACHE=""
LENGTH_GROUPING_WINDOW="64"

while (($#)); do
    case "$1" in
        --train_type) TRAIN_TYPE="$2"; shift 2 ;;
        --grpo_response_batch_size) GRPO_ARGS+=("$1" "$2"); shift 2 ;;
        --dynamic_sampling_initial_accept_rate|--dynamic_sampling_max_factor)
            GRPO_ARGS+=("$1" "$2"); shift 2 ;;
        --loss_type|--sync_vllm|--importance_sampling_level|--vllm_importance_sampling_cap|--thinking|--grpo_offload_model|--max_output_token|--num_generations|--temperature|--top_p|--top_k|--repetition_penalty|--reward_weights|--epsilon|--dynamic_sampling_oversample_factor|--vllm_gpu_memory_utilization|--vllm_max_model_len|--vllm_max_num_batched_tokens)
            GRPO_ARGS+=("$1" "$2"); shift 2 ;;
        --tp_size|--vllm_tensor_parallel_size) GRPO_ARGS+=(--vllm_tensor_parallel_size "$2"); shift 2 ;;
        --dynamic_sampling|--vllm_enforce_eager) GRPO_ARGS+=("$1" true); shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model_base) MODEL_BASE="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --total_timestamps) TOTAL_TIMESTAMPS="$2"; shift 2 ;;
        --lora_ckpt|--lora-ckpt) LORA_CKPT="$2"; shift 2 ;;
        --lora_ckpt_2) LORA_CKPT_2="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --validation_data) VALIDATION_DATA="$2"; shift 2 ;;
        --length_cache) LENGTH_CACHE="$2"; shift 2 ;;
        --length_grouping_window) LENGTH_GROUPING_WINDOW="$2"; shift 2 ;;
        --report_to) REPORT_TO="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --bs) BS="$2"; shift 2 ;;
        --accum_steps) ACCUM_STEPS="$2"; shift 2 ;;
        --epoch) EPOCH="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --save_steps) SAVE_STEPS="$2"; shift 2 ;;
        --max_pixels) MAX_PIXELS="$2"; shift 2 ;;
        --max_pixels_img) MAX_PIXELS_IMG="$2"; shift 2 ;;
        --min_pixels) MIN_PIXELS="$2"; shift 2 ;;
        --min_pixels_img) MIN_PIXELS_IMG="$2"; shift 2 ;;
        --min_frames) MIN_FRAMES="$2"; shift 2 ;;
        --max_frames) MAX_FRAMES="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --num_worker) NUM_WORKER="$2"; shift 2 ;;
        --prefetch_factor) PREFETCH_FACTOR="$2"; shift 2 ;;
        --warmup_ratio) WARMUP_RATIO="$2"; shift 2 ;;
        --lora_r) LORA_R="$2"; shift 2 ;;
        --lora_alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --shard_model) SHARD_MODEL=true; shift ;;
        --no_checkpoint) NO_CHECKPOINT=true; shift ;;
        --attn_implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
        --no_hdfs_upload) HDFS_UPLOAD=false; shift ;;
        --hdfs_output_dir) HDFS_OUTPUT_DIR="$2"; shift 2 ;;
        --train_llm) TRAIN_LLM=true; shift ;;
        --train_proj) TRAIN_PROJ=true; shift ;;
        --train_enc) TRAIN_ENC=true; shift ;;
        --train_audio) TRAIN_AUDIO=true; shift ;;
        --train_qformer) TRAIN_QFORMER=true; shift ;;
        --use_lora) USE_LORA=true; shift ;;
        --do_validation) DO_VALIDATION=true; shift ;;
        --use_rloo) USE_RLOO=true; shift ;;
        --grpo_num_sample) GRPO_NUM_SAMPLE="$2"; shift 2 ;;
        --planner_type) PLANNER_TYPE="$2"; shift 2 ;;
        --action_samples) ACTION_SAMPLES="$2"; shift 2 ;;
        --adapter_update) ADAPTER_UPDATE="$2"; shift 2 ;;
        --reward_func) REWARD_FUNC="$2"; shift 2 ;;
        --cua_kl_beta) CUA_KL_BETA="$2"; shift 2 ;;
        --cua_max_plan_tokens) CUA_MAX_PLAN_TOKENS="$2"; shift 2 ;;
        --cua_max_action_tokens) CUA_MAX_ACTION_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$MAX_PIXELS_IMG" ]] || MAX_PIXELS_IMG="$MAX_PIXELS"
[[ -n "$MODEL" && -n "$DATASET" ]] || { echo "--model and --dataset are required" >&2; exit 2; }
if [[ -n "$TOTAL_TIMESTAMPS" ]]; then
    [[ "$TOTAL_TIMESTAMPS" =~ ^[1-9][0-9]*$ ]] || { echo "--total_timestamps must be a positive integer" >&2; exit 2; }
    echo "Warning: --total_timestamps is a Qwen3-VL legacy option; Qwen3.5 uses one timestamp per video temporal patch and does not apply this limit." >&2
fi
if [[ "$DO_VALIDATION" == true ]]; then
    [[ -n "$VALIDATION_DATA" ]] || { echo "--do_validation requires --validation_data" >&2; exit 2; }
fi
export WANDB_PROJECT=thu_qwenvl
export WANDB_NAME="$RUN_NAME"
if [[ "$REPORT_TO" == wandb ]]; then
    wandb online
fi

materialize_hdfs() {
    local source="$1" kind="$2" name parent target
    if [[ "$source" != hdfs://* ]]; then
        printf '%s\n' "$source"
        return
    fi
    name="$(basename "${source%/}")"
    if [[ "$kind" == dataset ]]; then
        parent="$ROOT/dataset"
    elif [[ "$kind" == length_cache ]]; then
        parent="$ROOT/dataset/length_cache"
    elif [[ "$name" == checkpoint-* || "$name" == base ]]; then
        parent="$ROOT/output/$(basename "$(dirname "${source%/}")")"
    else
        parent="$ROOT/output"
    fi
    target="$parent/$name"
    if [[ ! -e "$target" ]]; then
        echo "Copying $source to $target" >&2
        PYTHONNOUSERSITE=1 python "$ROOT/scripts/hdfs_get.py" "$source" "$parent"
    fi
    printf '%s\n' "$target"
}

MODEL="$(materialize_hdfs "$MODEL" model)"
if [[ -n "$MODEL_BASE" ]]; then
    MODEL_BASE="$(materialize_hdfs "$MODEL_BASE" model)"
fi
if [[ -n "$LORA_CKPT" ]]; then
    LORA_CKPT="$(materialize_hdfs "$LORA_CKPT" model)"
fi
if [[ -n "$LORA_CKPT_2" ]]; then
    LORA_CKPT_2="$(materialize_hdfs "$LORA_CKPT_2" model)"
fi
DATASET="$(materialize_hdfs "$DATASET" dataset)"
if [[ -n "$VALIDATION_DATA" ]]; then
    VALIDATION_DATA="$(materialize_hdfs "$VALIDATION_DATA" dataset)"
fi
if [[ -n "$LENGTH_CACHE" ]]; then
    LENGTH_CACHE="$(materialize_hdfs "$LENGTH_CACHE" length_cache)"
fi
CONFIG_MODEL="$MODEL"
if [[ -f "$MODEL/adapter_config.json" ]]; then
    CONFIG_MODEL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' "$MODEL/adapter_config.json")"
    if [[ -n "$MODEL_BASE" ]]; then
        CONFIG_MODEL="$MODEL_BASE"
    else
        CONFIG_MODEL="$(materialize_hdfs "$CONFIG_MODEL" model)"
    fi
fi
[[ -n "$MODEL_BASE" ]] || MODEL_BASE="$MODEL"
if [[ "$MODEL_BASE" == "$MODEL" && "$CONFIG_MODEL" != "$MODEL" ]]; then
    MODEL_BASE="$CONFIG_MODEL"
fi
# Merge once per node before torchrun initializes distributed process groups.
if [[ "$TRAIN_TYPE" == "cua_grpo" && -f "$MODEL/adapter_config.json" ]]; then
    echo "CUA GRPO --model must be a full Qwen3.5 checkpoint; pass adapters with --lora_ckpt and --lora_ckpt_2" >&2
    exit 2
fi
if [[ "$TRAIN_TYPE" != "cua_grpo" && ( -n "$LORA_CKPT" || -f "$MODEL/adapter_config.json" ) ]]; then
    MERGE_ARGS=(--model "$MODEL" --model_base "$MODEL_BASE" --output_dir "output/$RUN_NAME")
    if [[ -n "$LORA_CKPT" ]]; then
        MERGE_ARGS+=(--lora_ckpt "$LORA_CKPT")
    fi
    python qwenvl/train/prepare_model.py "${MERGE_ARGS[@]}"
    MODEL="$ROOT/output/$RUN_NAME/base"
    MODEL_BASE="$MODEL"
    CONFIG_MODEL="$MODEL"
fi
FSDP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/qwenvl-fsdp.XXXXXX.json")"
trap 'rm -f -- "$FSDP_CONFIG"' EXIT
python - "$CONFIG_MODEL/config.json" "$SHARD_MODEL" "$NO_CHECKPOINT" "$FSDP_CONFIG" "$MODEL_TYPE" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    model_type = json.load(handle)["model_type"]

layer_classes = {
    "qwen3_5": ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"],
    "qwen3_5_moe": ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"],
}.get(model_type)
if layer_classes is None:
    raise SystemExit(f"Unsupported model_type: {model_type}")
requested = sys.argv[5]
if requested and requested not in (model_type, {"qwen3_5": "dense", "qwen3_5_moe": "moe"}[model_type]):
    raise SystemExit(f"--model_type {requested} disagrees with checkpoint model_type {model_type}")

config = {
    "version": 2,
    "reshard_after_forward": sys.argv[2] == "true",
    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "transformer_layer_cls_to_wrap": layer_classes + ["WhisperEncoderLayer", "BertLayer"],
    "activation_checkpointing": sys.argv[3] != "true",
    "cpu_ram_efficient_loading": False,
    "state_dict_type": "FULL_STATE_DICT",
}
with open(sys.argv[4], "w") as handle:
    json.dump(config, handle, indent=2)
PY

mkdir -p "output/$RUN_NAME"
HDFS_ARGS=()
if [[ "$HDFS_UPLOAD" == true ]]; then
    : "${HDFS_OUTPUT_DIR:?Set HDFS_OUTPUT_DIR when enabling HDFS upload}"
    HDFS_ARGS=(--hdfs_output_dir "$HDFS_OUTPUT_DIR")
fi
LAUNCH=(--nproc_per_node="${ARNOLD_WORKER_GPU:-8}" --nnodes="${ARNOLD_WORKER_NUM:-1}")
if [[ "${ARNOLD_WORKER_NUM:-1}" != 1 ]]; then
    LAUNCH+=(--node_rank="${ARNOLD_ID:-0}" --master_addr="${METIS_WORKER_0_HOST:?}" --master_port="${MASTER_PORT:-12396}")
fi

TRAIN_COMMAND=(torchrun "${LAUNCH[@]}" qwenvl/train/train_qwen.py \
    --train_type "$TRAIN_TYPE" \
    "${GRPO_ARGS[@]}" \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --dataset_use "$DATASET" \
    --tune_mm_llm "$TRAIN_LLM" \
    --tune_mm_mlp "$TRAIN_PROJ" \
    --tune_mm_vision "$TRAIN_ENC" \
    --tune_mm_audio "$TRAIN_AUDIO" \
    --tune_mm_qformer "$TRAIN_QFORMER" \
    --use_lora "$USE_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --bf16 true \
    --fsdp \
    --fsdp_config "$FSDP_CONFIG" \
    --gradient_checkpointing false \
    --output_dir "output/$RUN_NAME" \
    --num_train_epochs "$EPOCH" \
    --seed 2025 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$BS" \
    --gradient_accumulation_steps "$ACCUM_STEPS" \
    --max_pixels "$MAX_PIXELS_IMG" \
    --min_pixels "$MIN_PIXELS_IMG" \
    --video_max_frame_pixels "$MAX_PIXELS" \
    --video_min_frame_pixels "$MIN_PIXELS" \
    --video_max_frames "$MAX_FRAMES" \
    --video_min_frames "$MIN_FRAMES" \
    --base_interval "$INTERVAL" \
    --eval_strategy "$([[ "$TRAIN_TYPE" == sft && "$DO_VALIDATION" == true ]] && echo steps || echo no)" \
    --save_strategy steps \
    "${HDFS_ARGS[@]}" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$SAVE_STEPS" \
    --save_total_limit 10 \
    --learning_rate "$LR" \
    --weight_decay 0 \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --logging_nan_inf_filter false \
    --dataloader_num_workers "$NUM_WORKER" \
    --dataloader_prefetch_factor "$PREFETCH_FACTOR" \
    --run_name "$RUN_NAME" \
    --report_to "$REPORT_TO" \
    --remove_unused_columns false \
    --save_only_model true)

if [[ -n "$LENGTH_CACHE" ]]; then
    TRAIN_COMMAND+=(--length_cache "$LENGTH_CACHE" --length_grouping_window "$LENGTH_GROUPING_WINDOW")
fi
if [[ "$TRAIN_TYPE" == "sft" && "$DO_VALIDATION" == true ]]; then
    TRAIN_COMMAND+=(--do_validation true --validation_data "$VALIDATION_DATA")
fi
if [[ "$TRAIN_TYPE" == "cua_grpo" ]]; then
    TRAIN_COMMAND+=(
        --lora_ckpt "$LORA_CKPT" --lora_ckpt_2 "$LORA_CKPT_2"
        --grpo_num_sample "$GRPO_NUM_SAMPLE" --planner_type "$PLANNER_TYPE"
        --action_samples "$ACTION_SAMPLES" --adapter_update "$ADAPTER_UPDATE"
        --reward_func "$REWARD_FUNC" --use_rloo "$USE_RLOO"
        --cua_kl_beta "$CUA_KL_BETA"
        --cua_max_plan_tokens "$CUA_MAX_PLAN_TOKENS"
        --cua_max_action_tokens "$CUA_MAX_ACTION_TOKENS"
        --do_validation "$DO_VALIDATION")
    if [[ "$DO_VALIDATION" == true ]]; then
        TRAIN_COMMAND+=(--validation_data "$VALIDATION_DATA")
    fi
fi

"${TRAIN_COMMAND[@]}"
