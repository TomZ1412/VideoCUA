cd "$(cd "$(dirname "$0")" && pwd)"

DATASET="data_path"
MODEL="model_path"
MODEL_BASE="base_path"
LR=2e-5
BS=1
ACCUM_STEPS=1
RUN_NAME="qwen2vl-baseline"
DEEPSPEED=2
TRAIN_LLM=False
TRAIN_PROJ=False
TRAIN_ENC=False
TRAIN_AUDIO=False
TRAIN_QFORMER=False
EPOCH=1
FREEZE_TTT=False
FREEZE_LORA=False
MEMGROUPSIZE=0
WORKINGMEMSIZE=0
SEARCH_TYPE=none
SLOT_TYPE="none"
EMA_FACTOR=0.1
LAG_DISTANCES="0"
DIV_FACTOR=0.0
TOTAL_TIMESTAMPS=5000
PRETRAINED_TTT_LAYERS="-1"
GLOBAL_FACTOR=0.0
LOSS_FACTOR=0.0
SAVEMODE_STOPLAYER=0
SHRINK_SIZE=0
SHRINK_CONFIG="none"
ADAPTER_UPDATE="both"
REWARD_FUNC="sample"

MAX_PIXELS=176400
MIN_PIXELS=784
MAX_PIXELS_IMG=13107200
MIN_PIXELS_IMG=784

SAVE_STEPS=1000

MIN_FRAMES=64
MAX_FRAMES=128
INTERVAL=0.2

USE_LORA=False
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
LORA_CKPT=No
LORA_CKPT_2=No
DO_VALIDATION=False
VALIDATION_DATA=GUI_Net/data_processing/GUI_net_valid.json

TRAIN_TYPE=sft

NUM_WORKER=8
N_FULL_RES=0
FIXED_MEMORY_SIZE=0
FIXED_MEMORY_SIZE_AUDIO=0
STEPSIZE=0
TTT_TYPE=sim
CG_max_iter=0
TTT_HIDDEN_SIZE=4
TTT_NUM_HEADS=8
DISTILL_FACTOR=1.0
ALL_TTT_LAYERS="0"
ADVANTAGE_FACTOR=0
GRPO_NUM_SAMPLE=0
PLANNER_TYPE="none"
USE_RLOO=False
ACTION_SAMPLES=1

NO_AUDIO=False
MODEL_TYPE=moe
FSDP=full_shard
OFFLOAD=

mkdir -p output
mkdir -p dataset

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --accum_steps) ACCUM_STEPS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --deepspeed) DEEPSPEED="$2"; shift ;;
        --train_llm) TRAIN_LLM=True ;;
        --train_proj) TRAIN_PROJ=True ;;
        --train_enc) TRAIN_ENC=True ;;
        --train_audio) TRAIN_AUDIO=True ;;
        --train_qformer) TRAIN_QFORMER=True ;;
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --max_pixels_img) MAX_PIXELS_IMG="$2"; shift ;;
        --min_pixels_img) MIN_PIXELS_IMG="$2"; shift ;;
        --epoch) EPOCH="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --interval) INTERVAL="$2"; shift ;;
        --use_lora) USE_LORA=True ;;
        --lora_r) LORA_R="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --lora_ckpt_2) LORA_CKPT_2="$2"; shift ;;
        --train_type) TRAIN_TYPE="$2"; shift ;;
        --num_worker) NUM_WORKER="$2"; shift ;;
        --no_audio) NO_AUDIO=True ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --do_validation) DO_VALIDATION=True ;;
        --fixed_memory_size) FIXED_MEMORY_SIZE="$2"; shift ;;
        --stepsize) STEPSIZE="$2"; shift ;;
        --ttt_type) TTT_TYPE="$2"; shift ;;
        --cg_max_iter) CG_max_iter="$2"; shift ;;
        --validation_data) VALIDATION_DATA="$2"; shift ;;
        --ttt_hidden_size) TTT_HIDDEN_SIZE="$2"; shift ;;
        --ttt_num_heads) TTT_NUM_HEADS="$2"; shift ;;
        --fixed_memory_size_audio) FIXED_MEMORY_SIZE_AUDIO="$2"; shift ;;
        --distill_factor) DISTILL_FACTOR="$2"; shift ;;
        --fsdp) FSDP="$2"; shift ;;
        --freeze_ttt) FREEZE_TTT=True ;;
        --memgroupsize) MEMGROUPSIZE="$2"; shift ;;
        --workingmemsize) WORKINGMEMSIZE="$2"; shift ;;
        --search_type) SEARCH_TYPE="$2"; shift ;;
        --freeze_lora) FREEZE_LORA=True ;;
        --div_factor) DIV_FACTOR="$2"; shift ;;
        --slot_type) SLOT_TYPE="$2"; shift ;;
        --ema_factor) EMA_FACTOR="$2"; shift ;;
        --lag_distances) LAG_DISTANCES="$2"; shift ;;
        --all_ttt_layers) ALL_TTT_LAYERS="$2"; shift ;;
        --total_timestamps) TOTAL_TIMESTAMPS="$2"; shift ;;
        --pretrained_ttt_layers) PRETRAINED_TTT_LAYERS="$2"; shift ;;
        --global_factor) GLOBAL_FACTOR="$2"; shift ;;
        --loss_factor) LOSS_FACTOR="$2"; shift ;;
        --savemode_stoplayer) SAVEMODE_STOPLAYER="$2"; shift ;;
        --shrinksize) SHRINK_SIZE="$2"; shift ;;
        --shrink_config) SHRINK_CONFIG="$2"; shift ;;
        --n_full_res) N_FULL_RES="$2"; shift ;;
        --advantage_factor) ADVANTAGE_FACTOR="$2"; shift ;;
        --grpo_num_sample) GRPO_NUM_SAMPLE="$2"; shift ;;
        --planner_type) PLANNER_TYPE="$2"; shift ;;
        --use_rloo) USE_RLOO=True ;;
        --action_samples) ACTION_SAMPLES="$2"; shift ;;
        --adapter_update) ADAPTER_UPDATE="$2"; shift ;;
        --reward_func) REWARD_FUNC="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ $MODEL == "hdfs"* ]]; then
    if [[ $MODEL =~ checkpoint ]]; then
        mkdir -p output/$(basename $(dirname $MODEL))
        hdfs dfs get $MODEL output/$(basename $(dirname $MODEL))
        MODEL=output/$(basename $(dirname $MODEL))/$(basename $MODEL)
    else
        hdfs dfs get $MODEL output/
        MODEL=output/$(basename $MODEL)
    fi
fi

if [[ $LORA_CKPT == "hdfs"* ]]; then
    if [[ $LORA_CKPT =~ checkpoint ]]; then
        mkdir -p output/$(basename $(dirname $LORA_CKPT))
        hdfs dfs get $LORA_CKPT output/$(basename $(dirname $LORA_CKPT))
        LORA_CKPT=output/$(basename $(dirname $LORA_CKPT))/$(basename $LORA_CKPT)
    else
        hdfs dfs get $LORA_CKPT output/
        LORA_CKPT=output/$(basename $LORA_CKPT)
    fi
fi

if [[ $MODEL_BASE == "hdfs"* ]]; then
    hdfs dfs get $MODEL_BASE output/
    MODEL_BASE=output/$(basename $MODEL_BASE)
fi


if [[ $DATASET == "hdfs"* ]]; then
    hdfs dfs get $DATASET dataset/
    DATASET=dataset/$(basename $DATASET)
fi

GRAD_CHECKPOINT=True

NPROC_PER_NODE="${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-1}}"
NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${ARNOLD_ID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${METIS_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-12397}"
DEEPSPEED_CONFIG="scripts/zero${DEEPSPEED}.json"

if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
    echo "DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
    exit 1
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    qwenvl/train/train_qwen.py \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --model_name_or_path "$MODEL" \
        --dataset_use $DATASET \
        --tune_mm_vision $TRAIN_ENC \
        --tune_mm_mlp $TRAIN_PROJ \
        --tune_mm_llm $TRAIN_LLM \
        --bf16 \
        --output_dir output/$RUN_NAME \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BS \
        --gradient_accumulation_steps $ACCUM_STEPS \
        --max_pixels $MAX_PIXELS_IMG \
        --min_pixels $MIN_PIXELS_IMG \
        --video_max_frame_pixels $MAX_PIXELS \
        --video_min_frame_pixels $MIN_PIXELS \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps $SAVE_STEPS \
        --save_total_limit 10 \
        --learning_rate $LR \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 1310720 \
        --dataloader_num_workers $NUM_WORKER \
        --run_name $RUN_NAME \
        --report_to wandb \
        --gradient_checkpointing $GRAD_CHECKPOINT \
        --video_min_frames $MIN_FRAMES \
        --video_max_frames $MAX_FRAMES \
        --base_interval $INTERVAL \
        --model_base $MODEL_BASE \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_ckpt $LORA_CKPT \
        --train_type $TRAIN_TYPE \
        --tune_mm_audio $TRAIN_AUDIO \
        --tune_mm_qformer $TRAIN_QFORMER \
        --no_audio $NO_AUDIO \
        --use_modality_sampler True \
        --model_type $MODEL_TYPE \
        --fixed_memory_size $FIXED_MEMORY_SIZE \
        --stepsize $STEPSIZE \
        --ttt_type $TTT_TYPE \
        --cg_max_iter $CG_max_iter \
        --validation_data $VALIDATION_DATA \
        --ttt_hidden_size $TTT_HIDDEN_SIZE \
        --ttt_num_heads $TTT_NUM_HEADS \
        --do_validation $DO_VALIDATION \
        --fixed_memory_size_audio $FIXED_MEMORY_SIZE_AUDIO \
        --distill_factor $DISTILL_FACTOR \
        --freeze_ttt $FREEZE_TTT \
        --memgroupsize $MEMGROUPSIZE \
        --workingmemsize $WORKINGMEMSIZE \
        --freeze_lora $FREEZE_LORA \
        --search_type $SEARCH_TYPE \
        --ema_factor $EMA_FACTOR \
        --lag_distances $LAG_DISTANCES \
        --div_factor $DIV_FACTOR \
        --slot_type $SLOT_TYPE \
        --total_timestamps $TOTAL_TIMESTAMPS \
        --all_ttt_layers $ALL_TTT_LAYERS \
        --pretrained_ttt_layers $PRETRAINED_TTT_LAYERS \
        --global_factor $GLOBAL_FACTOR \
        --loss_factor $LOSS_FACTOR \
        --savemode_stoplayer $SAVEMODE_STOPLAYER \
        --shrinksize $SHRINK_SIZE \
        --shrinksize_config $SHRINK_CONFIG \
        --n_full_res $N_FULL_RES \
        --advantage_factor $ADVANTAGE_FACTOR \
        --grpo_num_sample $GRPO_NUM_SAMPLE \
        --planner_type $PLANNER_TYPE \
        --use_rloo $USE_RLOO \
        --action_samples $ACTION_SAMPLES \
        --lora_ckpt_2 $LORA_CKPT_2 \
        --adapter_update $ADAPTER_UPDATE \
        --reward_func $REWARD_FUNC \
        --save_only_model True
