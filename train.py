
import os
import wandb
import torch
import random
import argparse
import transformers
import numpy as np


from utils.loader import Loader
from utils.encoder import Encoder
from utils.metirc import compute_metrics
from utils.collator import DataCollatorForTraining

from dotenv import load_dotenv
from transformers import (AutoConfig, 
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments, 
)

def train(args):
    # -- Checkpoint 
    MODEL_NAME = args.PLM
    print('Model : %s' %MODEL_NAME)

    # -- Loading Dataset
    print('\nLoading Dataset')    
    loader = Loader(dir_path=args.data_dir, seed=args.seed)
    dset = loader.get() if args.full_training else loader.split()
    print(dset)

    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Config & Model
    print('\nLoading Config and Model')
    config =  AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 2
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config).to(device)

    # -- Tokenizing Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -- Encoding Dataset
    print('\nEncoding Dataset')
    column_names = dset.column_names if args.full_training else dset['train'].column_names
    encoder = Encoder(plm=MODEL_NAME, tokenizer=tokenizer, max_length=args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=args.num_proc, remove_columns=column_names)
    print(dset)

    if args.full_training == True :
        train_dset = dset
        eval_dset = None
        evaluation_strategy = 'no'
    else :
        train_dset = dset['train']
        eval_dset = dset['validation']
        eval_batch_size = args.train_batch_size * args.gradient_accumulation_steps
        evaluation_strategy = 'steps'

    # -- Training Argument for training & validation
    training_args = TrainingArguments(
        output_dir=args.output_dir,                                     # output directory
        overwrite_output_dir=True,                                      # overwrite output directory
        save_total_limit=5,                                             # number of total save model.
        save_steps=args.save_steps,                                     # model saving step.
        num_train_epochs=args.epochs,                                   # total number of training epochs
        learning_rate=args.lr,                                          # learning_rate
        evaluation_strategy=evaluation_strategy,                        # evaluation_strategy strategy
        eval_steps=args.save_steps,                                     # eval steps
        per_device_eval_batch_size=eval_batch_size,                     # eval batch_size
        per_device_train_batch_size=args.train_batch_size,              # batch size per device during training
        warmup_steps=args.warmup_steps,                                 # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,                                 # strength of weight decay
        logging_dir=args.logging_dir,                                   # directory for storing logs
        logging_steps=args.logging_steps,                               # log saving step.
        fp16=True,                                                      # fp16 flag
        gradient_accumulation_steps=args.gradient_accumulation_steps,   # gradient accumulation steps
        report_to='wandb'
    )

    # -- Collator
    collator = DataCollatorForTraining(tokenizer=tokenizer, max_length=args.max_length)

    # -- Optimizer & Scheduler
    epoch_steps = len(train_dset) // (args.train_batch_size * args.gradient_accumulation_steps)
    training_steps = epoch_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=training_steps,
    )

    # -- Trainer
    trainer = Trainer(
        model=model,                                                    # the instantiated 🤗 Transformers model to be trained
        args=training_args,                                             # training arguments, defined above
        train_dataset=train_dset,                                       # training dataset
        eval_dataset=eval_dset,                                         # eval dataset
        data_collator=collator,                                         # collator
        optimizers=(optimizer, scheduler),                               # optimizer, scheduler
        compute_metrics=compute_metrics                                 # define metrics function
    )

    # -- Training
    print('Training Strats')
    trainer.train()

def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    wandb_name = f"EP:{args.epochs}_LR:{args.lr}_BS:{train_batch_size}_WS:{args.warmup_steps}_WD:{args.weight_decay}"
    wandb.init(entity="sangha0411", project="kaggle - NBME", name=wandb_name, group=args.PLM)
    wandb.config.update(args)
    train(args)
    wandb.finish()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- directory
    parser.add_argument('--output_dir', default='exp', help='trained model output directory')
    parser.add_argument('--logging_dir', default='logs', help='logging directory')
    parser.add_argument('--data_dir', default='data', help='train data directory path')
    
    # -- plm
    parser.add_argument('--PLM', type=str, default='microsoft/deberta-large', help='model type (microsoft/deberta-large)')

    # -- Data Length
    parser.add_argument('--max_length', type=int, default=360, help='max length of tensor (default: 360)')
    parser.add_argument('--num_proc', type=int, default=4, help='the number of processor (default: 4)')

    # -- training arguments
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (default: 3e-5)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('--train_batch_size', type=int, default=2, help='train batch size (default: 2)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='strength of weight decay (default: 1e-3)')
    parser.add_argument('--warmup_steps', type=int, default=200, help='number of warmup steps for learning rate scheduler (default: 200)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient_accumulation_steps (default: 4)')

    # -- save & log
    parser.add_argument('--full_training', type=bool, default=False, help='full training flag (default: False)')
    parser.add_argument('--save_steps', type=int, default=500, help='model save steps (default: 500)')
    parser.add_argument('--logging_steps', type=int, default=100, help='training log steps (default: 100)')

    # -- Seed
    parser.add_argument('--seed', type=int, default=2, help='random seed (default: 2)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='path.env', help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)

