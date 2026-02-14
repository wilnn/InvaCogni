#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
OUTPUT_DIR="model3/noTB_no_dc_img_text_no_aug14_avgtext"
RUN_NAME="noTB_no_dc_img_text_no_aug14_avgtext"
REPORT_TO="wandb"
NUM_FOLD=10
DATASET_PATH="./dataset/taukadial/final_combined_dataset.csv"
Audio_PARENT_PATH="./dataset/taukadial/taukadial/train/"
IMAGE_PARENT_PATH="./dataset/taukadial/images/images/"
DECISION_THRESHOLD=0.5
WANDB_PROJECT_NAME='InvaCogni'
MAX_DATASET_SIZE=-1 # negative to use the entire dataset
BATCH_SIZE=4
NUM_EPOCHS=14
AUDIO_FFN="[[512, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu']"
GENDER_DOMAIN_CLASSIFIER_FFN="[[512, 3072], 'gelu', [3072, 512], 'gelu', [512, 1]]"
LANGUAGE_DOMAIN_CLASSIFIER_FFN="[[1280, 3072], 'gelu', [3072, 768], 'gelu', [768, 1]]"
TASK_CLASSIFIER_FFN="[[1536, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu', 'dropout-0.3', [768, 384], 'gelu', [384, 1]]"
CROSS_ATTENTION_FFN="[[768, 3072], 'gelu', 'dropout-0.5', [3072, 768], 'gelu']"
ATTENTION_DROPOUT=0.35
NUM_ATTENTION_HEAD=8
model_class="InvaCogni_no_TB"
#--remove_punc_in_text \

accelerate launch train.py \
            --train_image_encoder \
            --train_text_encoder \
            --train_audio_encoder \
            --model_class="$model_class" \
            --audio_FFN="$AUDIO_FFN" \
            --gender_domain_classifier_FFN="$GENDER_DOMAIN_CLASSIFIER_FFN" \
            --language_domain_classifier_FFN="$LANGUAGE_DOMAIN_CLASSIFIER_FFN" \
            --task_classifier_FFN="$TASK_CLASSIFIER_FFN" \
            --cross_attn_FFN="$CROSS_ATTENTION_FFN" \
            --attention_dropout="$ATTENTION_DROPOUT" \
            --num_attention_heads="$NUM_ATTENTION_HEAD" \
            --num_fold=$NUM_FOLD \
            --dataset_path=$DATASET_PATH \
            --audio_parent_path=$Audio_PARENT_PATH \
            --image_parent_path=$IMAGE_PARENT_PATH \
            --pad_token=0 \
            --decision_threshold=$DECISION_THRESHOLD \
            --max_dataset_size=$MAX_DATASET_SIZE \
            --wandb_project_name=$WANDB_PROJECT_NAME \
            --include_for_metrics "inputs" "loss" \
            --eval_strategy="steps" \
            --eval_steps=0.026 \
            --save_strategy="best" \
            --load_best_model_at_end \
            --metric_for_best_model="avg_f1_bal_acc" \
            --save_total_limit=1 \
            --logging_strategy="steps" \
            --logging_steps=6 \
            --learning_rate=2e-5 \
            --weight_decay=2e-2 \
            --report_to=$REPORT_TO \
            --output_dir=$OUTPUT_DIR \
            --run_name=$RUN_NAME \
            --per_device_train_batch_size=$BATCH_SIZE \
            --per_device_eval_batch_size=$BATCH_SIZE \
            --do_train \
            --do_eval \
            --num_train_epochs=$NUM_EPOCHS \
            --dataloader_num_workers=2 \
            


