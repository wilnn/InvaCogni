#!/bin/bash

OUTPUT_DIR="model/test"
REPORT_TO="wandb"
RUN_NAME="test"
NUM_FOLD=10
DATASET_PATH="./dataset/combined_dataset.csv"
Audio_PARENT_PATH="./dataset/taukadial/train/"
IMAGE_PARENT_PATH="./dataset/images/images/"
DECISION_THRESHOLD=0.5
WANDB_PROJECT_NAME='InvaCogni'
MAX_DATASET_SIZE=-1 # negative to use the entire dataset
export CUDA_VISIBLE_DEVICES=2
BATCH_SIZE=4
NUM_EPOCHS=10
#--dc_gender \
#--dc_language \
accelerate launch train.py \
            --num_fold=$NUM_FOLD \
            --train_audio_encoder \
            --train_text_encoder \
            --dataset_path=$DATASET_PATH \
            --audio_parent_path=$Audio_PARENT_PATH \
            --image_parent_path=$IMAGE_PARENT_PATH \
            --pad_token=0 \
            --decision_threshold=$DECISION_THRESHOLD \
            --max_dataset_size=$MAX_DATASET_SIZE \
            --wandb_project_name=$WANDB_PROJECT_NAME \
            --include_for_metrics "inputs" "loss" \
            --eval_strategy="epoch" \
            --save_strategy="best" \
            --load_best_model_at_end \
            --metric_for_best_model="avg_f1_bal_acc" \
            --save_total_limit=2 \
            --report_to=$REPORT_TO \
            --output_dir=$OUTPUT_DIR \
            --run_name=$RUN_NAME \
            --per_device_train_batch_size=$BATCH_SIZE \
            --per_device_eval_batch_size=$BATCH_SIZE \
            --do_train \
            --do_eval \
            --learning_rate=2e-5 \
            --weight_decay=1e-2 \
            --num_train_epochs=$NUM_EPOCHS \
            --dataloader_num_workers=2 \
            --logging_strategy="steps" \
            --logging_steps=6 \
            


