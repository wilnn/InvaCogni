@echo off

set CUDA_VISIBLE_DEVICES=0
set OUTPUT_DIR=model/test2_no_text_moreeval12_augIMGAUDIO
set RUN_NAME=test2_no_text_moreeval12_augIMGAUDIO
set REPORT_TO=wandb
set NUM_FOLD=10
set DATASET_PATH=./dataset/combined_dataset.csv
set Audio_PARENT_PATH=./dataset/taukadial/train/
set IMAGE_PARENT_PATH=./dataset/images/images/
set DECISION_THRESHOLD=0.5
set WANDB_PROJECT_NAME=InvaCogni
set MAX_DATASET_SIZE=-1
set BATCH_SIZE=4
set NUM_EPOCHS=12

:: Complex FFN strings (keep as-is; will be passed quoted to the script)
set AUDIO_FFN=[[512, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu']
set GENDER_DOMAIN_CLASSIFIER_FFN=[[512, 3072], 'gelu', [3072, 512], 'gelu', [512, 1]]
set LANGUAGE_DOMAIN_CLASSIFIER_FFN=[[1280, 3072], 'gelu', [3072, 768], 'gelu', [768, 1]]
set TASK_CLASSIFIER_FFN=[[1536, 3072], 'gelu', 'dropout-0.3', [3072, 768], 'gelu', 'dropout-0.3', [768, 384], 'gelu', [384, 1]]
set CROSS_ATTENTION_FFN=[[768, 3072], 'gelu', 'dropout-0.5', [3072, 768], 'gelu']
set ATTENTION_DROPOUT=0.35
set NUM_ATTENTION_HEAD=8

:: Run training via accelerate
accelerate launch train.py ^
    --aug_img ^
    --aug_audio ^
    --train_audio_encoder ^
    --audio_FFN "%AUDIO_FFN%" ^
    --gender_domain_classifier_FFN "%GENDER_DOMAIN_CLASSIFIER_FFN%" ^
    --language_domain_classifier_FFN "%LANGUAGE_DOMAIN_CLASSIFIER_FFN%" ^
    --task_classifier_FFN "%TASK_CLASSIFIER_FFN%" ^
    --cross_attn_FFN "%CROSS_ATTENTION_FFN%" ^
    --attention_dropout %ATTENTION_DROPOUT% ^
    --num_attention_heads %NUM_ATTENTION_HEAD% ^
    --num_fold %NUM_FOLD% ^
    --dataset_path %DATASET_PATH% ^
    --audio_parent_path %Audio_PARENT_PATH% ^
    --image_parent_path %IMAGE_PARENT_PATH% ^
    --pad_token 0 ^
    --decision_threshold %DECISION_THRESHOLD% ^
    --max_dataset_size %MAX_DATASET_SIZE% ^
    --wandb_project_name %WANDB_PROJECT_NAME% ^
    --include_for_metrics "inputs" "loss" ^
    --eval_strategy "steps" ^
    --eval_steps 0.026 ^
    --save_strategy "best" ^
    --load_best_model_at_end ^
    --metric_for_best_model "avg_f1_bal_acc" ^
    --save_total_limit 1 ^
    --logging_strategy "steps" ^
    --logging_steps 6 ^
    --learning_rate 2e-5 ^
    --weight_decay 2e-2 ^
    --report_to "%REPORT_TO%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --run_name "%RUN_NAME%" ^
    --per_device_train_batch_size %BATCH_SIZE% ^
    --per_device_eval_batch_size %BATCH_SIZE% ^
    --do_train ^
    --do_eval ^
    --num_train_epochs %NUM_EPOCHS% ^
