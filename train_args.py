from transformers import TrainingArguments
from dataclasses import dataclass

@dataclass
class trainArgs(TrainingArguments):
    num_fold: int = 10
    train_text_encoder: bool = False # the encoder part of the text encoder
                                #(the pooler layer is always trained)
    train_audio_encoder: bool = False
    dataset_path: str = "./dataset/combined_dataset.csv"
    audio_parent_path: str ="./dataset/taukadial/train/",
    image_parent_path: str ="./dataset/images/images/",
    pad_token: int = 0
    dc_gender: bool = False
    dc_language: bool = False
    decision_threshold: float = 0.5
    wandb_project_name: str = 'InvaCogni'
    max_dataset_size: int = -1 # -1 means use full size.
                            # > 0 means dataset of that many samples
    aug_audio: bool = False
    aug_img: bool = False
    remove_punc_in_text: bool = False