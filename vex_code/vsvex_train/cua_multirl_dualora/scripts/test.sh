cd "$(cd $(dirname $0); pwd)/.."

DATASET=${DATASET:-/path/to/test_dataset.json}
MODEL=${MODEL:-/path/to/Qwen3-VL-8B-Instruct}
MODEL_BASE=${MODEL_BASE:-/path/to/Qwen3-VL-8B-Instruct}
LR=2e-5
BS=1
ACCUM_STEPS=1
RUN_NAME="qwen2vl-baseline"
DEEPSPEED=2
TRAIN_LLM=False
TRAIN_PROJ=False
TRAIN_ENC=False
EPOCH=1

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"
wandb login "$WANDB_API_KEY"
export WANDB_PROJECT=thu_qwenvl # llava-video-ours
export WANDB_NAME=$RUN_NAME
wandb online

MAX_PIXELS=176400
MIN_PIXELS=784

SAVE_STEPS=1000

MIN_FRAMES=4
MAX_FRAMES=128
INTERVAL=0.2

USE_LORA=False
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
LORA_CKPT=No
NUM_SAMPLE=1
DO_SAMPLE=False
NO_AUDIO=False

MODEL_TYPE=moe

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
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
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
        --do_sample) DO_SAMPLE=True ;;
        --num_sample) NUM_SAMPLE="$2"; shift ;;
        --no_audio) NO_AUDIO=True ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
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
else
    cp $DATASET dataset/
    DATASET=dataset/$(basename $DATASET)
fi

python3 scripts/split_data.py $DATASET $((ARNOLD_WORKER_GPU*ARNOLD_WORKER_NUM)) dataset/

HOST_ADDR=METIS_WORKER_${ARNOLD_ID}_HOST

# ${ARNOLD_WORKER_GPU}
for i in $(seq 0 $((${ARNOLD_WORKER_GPU}-1))); do
    CUDA_VISIBLE_DEVICES=$i torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="${!HOST_ADDR}" --master_port=$((12800 + i + 8 * ARNOLD_ID)) \
        qwenvl/train/train_qwen.py \
            --model_base $MODEL_BASE \
            --run_test True \
            --pred_rank $((ARNOLD_WORKER_GPU * ARNOLD_ID + i)) \
            --deepspeed scripts/zero${DEEPSPEED}.json \
            --model_name_or_path "$MODEL" \
            --dataset_use dataset/$((ARNOLD_WORKER_GPU * ARNOLD_ID + i)).json \
            --tune_mm_vision $TRAIN_ENC \
            --tune_mm_mlp $TRAIN_PROJ \
            --tune_mm_llm $TRAIN_LLM \
            --bf16 \
            --output_dir output/test \
            --num_train_epochs $EPOCH \
            --per_device_train_batch_size $BS \
            --gradient_accumulation_steps $ACCUM_STEPS \
            --max_pixels $MAX_PIXELS \
            --min_pixels $MIN_PIXELS \
            --video_max_frame_pixels $MAX_PIXELS \
            --video_min_frame_pixels $MIN_PIXELS \
            --eval_strategy "no" \
            --save_strategy "steps" \
            --save_steps $SAVE_STEPS \
            --save_total_limit 5 \
            --learning_rate $LR \
            --weight_decay 0 \
            --warmup_ratio 0.03 \
            --max_grad_norm 1 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --model_max_length 131072 \
            --gradient_checkpointing True \
            --dataloader_num_workers 8 \
            --run_name $RUN_NAME \
            --report_to wandb \
            --video_min_frames $MIN_FRAMES \
            --video_max_frames $MAX_FRAMES \
            --base_interval $INTERVAL \
            --use_lora $USE_LORA \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --lora_ckpt $LORA_CKPT \
            --num_sample $NUM_SAMPLE \
            --do_sample $DO_SAMPLE \
            --no_audio $NO_AUDIO \
            --model_type $MODEL_TYPE &
done

wait
