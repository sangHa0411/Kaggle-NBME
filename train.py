
import os
import wandb
import torch
import random
import numpy as np

from utils.loader import Loader
from utils.encoder import Encoder
from utils.metirc import compute_metrics
from utils.collator import DataCollatorWithPadding
from utils.preprocessor import process_features, clean_spaces
from model.model import DebertaForTokenClassification

from datasets import Dataset
from sklearn.model_selection import StratifiedKFold

from dotenv import load_dotenv
from transformers import (AutoConfig, 
    AutoTokenizer,
    HfArgumentParser,
    Trainer, 
)

from arguments import (ModelArguments, 
    DataArguments, 
    MyTrainingArguments, 
    LoggingArguments
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading Dataset
    print('\nLoading Dataset')    
    loader = Loader()
    df = loader.load(dir_path=data_args.dir_path)

    df['feature_text'] = df['feature_text'].apply(process_features)
    df['pn_history'] = df['pn_history'].apply(clean_spaces)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.remove_columns(["id", "pn_num", "feature_num", "annotation", "__index_level_0__"])

    case_num = dataset['case_num']

    # -- Config & Model
    print('\nLoading Config and Model')
    config =  AutoConfig.from_pretrained(model_args.PLM)
    config.num_labels = 1

    # -- Tokenizing Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)

    # -- Encoding Dataset
    print('\nEncoding Dataset')
    encoder = Encoder(tokenizer=tokenizer, max_length=data_args.max_length)
    dataset = dataset.map(encoder, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=dataset.column_names)
    print(dataset)

    # -- Collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    # -- K-fold
    skf = StratifiedKFold(n_splits=training_args.fold_size, shuffle=True)

    load_dotenv(dotenv_path=logging_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    for i, (train_ids, eval_ids) in enumerate(skf.split(dataset, case_num)):        
        train_dataset = dataset.select(train_ids.tolist()).shuffle(training_args.seed)
        eval_dataset = dataset.select(eval_ids.tolist()).shuffle(training_args.seed)

        model = DebertaForTokenClassification.from_pretrained(model_args.PLM, config=config)

        # -- Trainer
        trainer = Trainer(
            model=model,                                                    # the instantiated 🤗 Transformers model to be trained
            args=training_args,                                             # training arguments, defined above
            train_dataset=train_dataset,                                    # training dataset
            eval_dataset=eval_dataset,                                      # eval dataset
            data_collator=collator,                                         # collator
            compute_metrics=compute_metrics                                 # define metrics function
        )

        # -- Training
        print('Training Strats')

        if training_args.do_train :
            train_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            lr = training_args.learning_rate
            epochs = training_args.num_train_epochs
            warmup_steps = training_args.warmup_steps
            weight_decay = training_args.weight_decay
            wandb_name = f"EP:{epochs}_LR:{lr}_BS:{train_batch_size}_WS:{warmup_steps}_WD:{weight_decay}_fold{i+1}"
            
            group_name = model_args.PLM if training_args.do_eval else model_args.PLM + '-validation'
            wandb.init(entity="sangha0411", project="kaggle-NBME", name=wandb_name, group=group_name)
            wandb.config.update(training_args)
                
            trainer.train()
            trainer.save_model(os.path.join(model_args.save_path, f'fold{i+1}'))
            trainer.evaluate()
            
            wandb.finish()

        if training_args.do_eval :
            break

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    main()

