#!/bin/bash

OUTPUT_DIR="model/gen_lang"
REPORT_TO="wandb"
RUN_NAME="gen_lang"
NUM_FOLD = 10
DATASET_PATH = "./dataset/combined_dataset.csv"
Audio_PARENT_PATH ="./dataset/taukadial/train/"
IMAGE_PARENT_PATH ="./dataset/images/images/"
DECISION_THRESHOLD = 0.5
WANDB_PROJECT_NAME = 'InvaCogni'
MAX_DATASET_SIZE = -1 # negative to use the entire dataset
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=4
NUM_EPOCHS=10

accelerate launch train.py \
            --num_fold=$NUM_FOLD
            --dataset_path=$DATASET_PATH \
            --audio_parent_path=$Audio_PARENT_PATH, \
            --image_parent_path=$IMAGE_PARENT_PATH, \
            --pad_token=0 \
            --dc_gender \
            --dc_language \
            --decision_threshold=$DECISION_THRESHOLD \
            --wandb_project_name=$WANDB_PROJECT_NAME \
            --report_to=$REPORT_TO \
            --max_dataset_size=$MAX_DATASET_SIZE \
            --output_dir=$OUTPUT_DIR \
            --run_name=$RUN_NAME \
            --do_train \
            --per_device_train_batch_size=$BATCH_SIZE \
            --learning_rate=2e-5 \
            --weight_decay=1e-2 \
            --num_train_epochs=$NUM_EPOCHS \
            --dataloader_num_workers=2 \
            --logging_strategy="steps" \
            --logging_steps=6 \
            --save_strategy="no" \


