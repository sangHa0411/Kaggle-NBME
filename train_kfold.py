
import os
import wandb
import torch
import random
import argparse
import numpy as np

from utils.loader import Loader
from utils.metirc import compute_metrics
from utils.preprocessor import Preprocessor
from utils.collator import DataCollatorForTraining

from dotenv import load_dotenv
from transformers import (AutoTokenizer, 
    AutoConfig, 
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments, 
)

def train(args):
    # -- Checkpoint 
    MODEL_NAME = args.PLM
    print('Model : %s' %MODEL_NAME)

    # -- Loading Dataset
    print('\nLoad Dataset')    
    loader = Loader(dir_path=args.dir_path, validation_ratio=args.eval_ratio, seed=args.seed)
    dset = loader.get_total()
        
    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    # -- Tokenizing Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -- Preprocessing Dataset
    print('\nPreprocess Dataset')
    preprocessor = Preprocessor(tokenizer=tokenizer, max_length=args.max_length)
    dset = dset.map(preprocessor, batched=True, num_proc=args.num_proc, remove_columns=dset.column_names)
   
    # -- Config
    print('\nLoad Config and Model')
    config =  AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 2

    # -- Collator
    collator = DataCollatorForTraining(tokenizer=tokenizer, max_length=args.max_length)

    gap = int(len(dset) / args.k_fold)
    train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    wandb_name = f"epochs:{args.epochs}_learning_rate:{args.lr}_batch_size:{train_batch_size}_warmup_steps:{args.warmup_steps}_weight_decay:{args.weight_decay}"    
    for i in range(args.k_fold) :

        # -- Model
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config).to(device)
        
        # -- Wandb 
        group_name = args.PLM + '_k-fold'
        wandb_name = f"epochs:{args.epochs}_learning_rate:{args.lr}_batch_size:{args.train_batch_size}_warmup_steps:{args.warmup_steps}_weight_decay:{args.weight_decay}_k-fold:{i+1}" 
        wandb.init(entity="sangha0411", project="kaggle - NBME", name=wandb_name, group=group_name)
        wandb.config.update(args)
    
        # -- Training & Validation index
        total_size = len(dset)
        total_ids = list(range(total_size))
        del_ids = list(range(i*gap, (i+1)*gap))
        training_ids = set(total_ids) - set(del_ids)

        # -- Training & Validation datasets
        training_dset = dset.select(list(training_ids))
        eval_dset = dset.select(del_ids)

        # -- Output & Logging directory
        if os.path.exists(args.output_dir) == False:
            os.makedirs(args.output_dir)
        output_dir = os.path.join(args.output_dir, 'k-fold-' + str(i+1))
        logging_dir = args.logging_dir + '_' + str(i+1)

         # -- Training Argument
        training_args = TrainingArguments(
            output_dir=output_dir,                              # output directory
            overwrite_output_dir=True,                          # overwrite output directory
            save_total_limit=5,                                 # number of total save model.
            save_steps=args.save_steps,                         # model saving step.
            num_train_epochs=args.epochs,                       # total number of training epochs
            learning_rate=args.lr,                              # learning_rate
            per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.eval_batch_size,    # batch size for evaluation
            warmup_steps=args.warmup_steps,                     # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,                     # strength of weight decay
            logging_dir=logging_dir,                            # directory for storing logs
            logging_steps=args.logging_steps,                   # log saving step.
            evaluation_strategy=args.eval_strategy,             # evaluation strategy to adopt during training
            eval_steps=args.eval_steps,                         # evaluation step.
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            load_best_model_at_end = True,
            report_to='wandb'
        )

        # -- Trainer
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=training_dset,         # training dataset
            eval_dataset=eval_dset,              # evaluation dataset
            data_collator=collator,              # collator
            compute_metrics=compute_metrics      # define metrics function
        )

        # -- Training
        print('%dth Training Strats' %(i+1))
        trainer.train()
        wandb.finish()

def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)
    train(args)

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
    parser.add_argument('--output_dir',  type=str, default='./exp', help='trained model output directory')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='logging directory')
    parser.add_argument('--dir_path', type=str, default='./data/', help='train data directory path')

    parser.add_argument('--k_fold', type=int, default=5, help='train data directory path')
    
    # -- plm
    parser.add_argument('--PLM', type=str, default='roberta-large', help='model type (default: roberta-large)')

    # -- Data Length
    parser.add_argument('--max_length', type=int, default=512, help='max length of tensor (default: 512)')
    parser.add_argument('--num_proc', type=int, default=4, help='the number of processor (default: 4)')

    # -- training arguments
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (default: 3e-5)')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train (default: 3)')
    parser.add_argument('--train_batch_size', type=int, default=4, help='train batch size (default: 4)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='strength of weight decay (default: 1e-3)')
    parser.add_argument('--warmup_steps', type=int, default=200, help='number of warmup steps for learning rate scheduler (default: 200)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='gradient_accumulation_steps (default: 2)')

    # -- validation arguments
    parser.add_argument('--eval_ratio', type=float, default=0.2, help='evaluation ratio (default: 0.2)')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='eval batch size (default: 8)')
    parser.add_argument('--eval_strategy', type=str, default='steps', help='evaluation strategy to adopt during training, steps or epoch (default: steps)')
    
    # -- save & log
    parser.add_argument('--save_steps', type=int, default=500, help='model save steps')
    parser.add_argument('--logging_steps', type=int, default=100, help='training log steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='evaluation steps')

    # -- Seed
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='./path.env', help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)
