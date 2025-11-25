#!/bin/bash

OUTPUT_DIR="model/test2_05_01_3e-2"
REPORT_TO="wandb"
RUN_NAME="test2_dropout0.05_01"
NUM_FOLD=10
DATASET_PATH="./dataset/combined_dataset.csv"
Audio_PARENT_PATH="./dataset/taukadial/train/"
IMAGE_PARENT_PATH="./dataset/images/images/"
DECISION_THRESHOLD=0.5
WANDB_PROJECT_NAME='InvaCogni'
MAX_DATASET_SIZE=-1 # negative to use the entire dataset
export CUDA_VISIBLE_DEVICES=5
BATCH_SIZE=4
NUM_EPOCHS=12
AUDIO_FFN="[[512, 3072], 'gelu', 'dropout-0.1', [3072, 768], 'gelu']"
GENDER_DOMAIN_CLASSIIFER_FFN="[[512, 3072], 'gelu', [3072, 512], 'gelu', [512, 1]]"
LANGUAGE_DOMAIN_CLASSIFIER_FFN="[[1280, 3072], 'gelu', [3072, 768], 'gelu', [768, 1]]"
TASK_CLASSIIFER_FFN="[[1536, 3072], 'gelu', 'dropout-0.05', [3072, 768], 'gelu', 'dropout-0.05', [768, 384], 'gelu', [384, 1]]"
CROSS_ATTENTION_FFN='[[768, 3072], "gelu", "dropout-0.1", [3072, 768], "gelu"]'
#--dc_gender \
#--dc_language \
#--audio_FFN="$AUDIO_FFN" \
#--gender_domain_classifier_FFN="$GENDER_DOMAIN_CLASSIIFER_FFN" \
#--language_domain_classifier_FFN="$LANGUAGE_DOMAIN_CLASSIFIER_FFN" \
#--task_classifier_FFN="$TASK_CLASSIIFER_FFN" \
#--cross_attention_FFN="$CROSS_ATTENTION_FFN" \

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
            --save_total_limit=1 \
            --report_to=$REPORT_TO \
            --output_dir=$OUTPUT_DIR \
            --run_name=$RUN_NAME \
            --per_device_train_batch_size=$BATCH_SIZE \
            --per_device_eval_batch_size=$BATCH_SIZE \
            --do_train \
            --do_eval \
            --learning_rate=2e-5 \
            --weight_decay=3e-2 \
            --num_train_epochs=$NUM_EPOCHS \
            --dataloader_num_workers=2 \
            --logging_strategy="steps" \
            --logging_steps=6 \
            


