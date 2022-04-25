from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class ModelArguments : 
    PLM: str = field(
        default="microsoft/deberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    save_path: str = field(
        default="./checkpoints",
        metadata={
            "help": "Path to save checkpoint from fine tune model"
        },
    )
    
@dataclass
class DataArguments:
    max_length: int = field(
        default=512,
        metadata={
            "help": "Max length of input sequence"
        },
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={
            "help": "The number of preprocessing workers"
        }
    )
    dir_path: str = field(
        default='./data',
        metadata={
            "help": "Path to data directory"
        }
    )

    
@dataclass
class MyTrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(
        default='wandb',
    )
    fold_size : Optional[int] = field(
        default=5,
        metadata={"help" : "The number of folds"}
    )
   
@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )
    project_name: Optional[str] = field(
        default="Kaggle",
        metadata={"help": "project name"},
    )
