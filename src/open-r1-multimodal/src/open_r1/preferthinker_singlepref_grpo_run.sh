#!/bin/bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"


DATA_PATHS="/path/to/your/dataset.jsonl"
IMAGE_FOLDERS="/path/to/your/images"

MODEL_PATH="/path/to/your/model/Qwen2.5-VL-7B-Instruct"
SBERT_MODEL_PATH="/path/to/your/model/all-MiniLM-L6-v2"
DREAMSIM_CACHE_DIR="/path/to/your/dreamsim_cache"

DEEPSPEED_CONFIG="/path/to/your/deepspeed_config/zero3.json"

export EXP_NAME="GRPO_Single_Preference_Training"
TASK_TYPE="prefer_eval"


OUTPUT_DIR="${REPO_HOME}/checkpoints/rl/${EXP_NAME}"
GENERATED_IMAGE_DIR="${REPO_HOME}/outputs/generated_images"
FLUX_SERVER_DIR="${REPO_HOME}/outputs/flux_server_data"

cd "${REPO_HOME}"

mkdir -p "${REPO_HOME}/runs/${EXP_NAME}/log"
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/training_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

echo "Task: $TASK_TYPE"
echo "Data paths: $DATA_PATHS"
echo "Model path: $MODEL_PATH"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    "${REPO_HOME}/src/open_r1/preferthinker_singlepref_grpo.py" \
    --use_vllm False \
    --output_dir "$OUTPUT_DIR" \
    --resume_from_checkpoint True \
    --model_name_or_path "$MODEL_PATH" \
    --sbert_model_path "$SBERT_MODEL_PATH" \
    --dreamsim_cache_dir "$DREAMSIM_CACHE_DIR" \
    --generated_image_dir "$GENERATED_IMAGE_DIR" \
    --flux_server_dir "$FLUX_SERVER_DIR" \
    --data_file_paths "$DATA_PATHS" \
    --image_folders "$IMAGE_FOLDERS" \
    --is_reward_customized_from_vlm_module False \
    --task_type "$TASK_TYPE" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation sdpa \
    --run_name "${EXP_NAME}" \
    --save_steps 150 \
    --num_generations 2 \
    --max_completion_length 2048 \
    --reward_funcs accuracy prefer_attributes non-prefer_attributes format \
    --beta 0.04 \
    --epsilon_high 0.28 \
    --report_to tensorboard \
    --dataset-name not_used \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --reward_method all_match \
    --prefer_attributes_reward_method prefer_attributes \
    --non_prefer_attributes_reward_method non_prefer_attributes

echo "Training completed for ${EXP_NAME}"