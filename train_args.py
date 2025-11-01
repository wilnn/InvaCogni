from transformers import TrainingArguments
from dataclasses import dataclass

@dataclass
class trainArgs(TrainingArguments):
    num_fold: int = 5