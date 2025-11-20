@echo off
:: Set environment variables
set OUTPUT_DIR=model\no_dc
set REPORT_TO=wandb
set RUN_NAME=no_dc
set NUM_FOLD=2
set DATASET_PATH=.\dataset\combined_dataset.csv
set AUDIO_PARENT_PATH=.\dataset\taukadial\train\
set IMAGE_PARENT_PATH=.\dataset\images\images\
set DECISION_THRESHOLD=0.5
set WANDB_PROJECT_NAME=InvaCogni
set MAX_DATASET_SIZE=10
set CUDA_VISIBLE_DEVICES=0
set BATCH_SIZE=4
set NUM_EPOCHS=2

:: Launch training script
accelerate launch train.py ^
    --num_fold=%NUM_FOLD% ^
    --dataset_path=%DATASET_PATH% ^
    --audio_parent_path=%AUDIO_PARENT_PATH% ^
    --image_parent_path=%IMAGE_PARENT_PATH% ^
    --pad_token=0 ^
    --decision_threshold=%DECISION_THRESHOLD% ^
    --wandb_project_name=%WANDB_PROJECT_NAME% ^
    --report_to=%REPORT_TO% ^
    --max_dataset_size=%MAX_DATASET_SIZE% ^
    --output_dir=%OUTPUT_DIR% ^
    --run_name=%RUN_NAME% ^
    --do_train ^
    --per_device_train_batch_size=%BATCH_SIZE% ^
    --learning_rate=2e-5 ^
    --weight_decay=1e-2 ^
    --num_train_epochs=%NUM_EPOCHS% ^
    --dataloader_num_workers=2 ^
    --logging_strategy="steps"^
    --logging_steps=6^
    --save_strategy="no"^
