from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

import torch

import polars as pl
import wandb
import os, gc
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

exp_name = 'sep_token'
bs = 4
lr = 1e-5
epochs = 3
wd = 0.01
resize = False

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'microsoft/deberta-v3-large'

df = pl.read_parquet('data/train.parquet')
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    responses = [f"{r[0]}{sep_token}{r[1]}{sep_token}{r[2]}{sep_token}" for r in chat_list]

    return {'prompt': ''.join(responses)}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    lambda x: tokenizer(x['prompt'], max_length=1024, truncation=True), 
    batched=True,
    remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
)

def get_fold(fold_num):
    return DatasetDict({
        "train": tok_ds.filter(lambda x: x['fold'] != fold_num), 
        "test": tok_ds.filter(lambda x: x['fold'] == fold_num)
    })

def get_trainer(dds):
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    args = TrainingArguments(
        f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}',
        learning_rate=lr, warmup_ratio=0., 
        # gradient_accumulation_steps=accum,
        lr_scheduler_type='cosine', 
        bf16=True,
        bf16_full_eval=True,
        optim='adamw_bnb_8bit',
        evaluation_strategy="epoch", 
        logging_steps=1,
        per_device_train_batch_size=bs, 
        per_device_eval_batch_size=bs,
        greater_is_better=False, 
        group_by_length=True,
        num_train_epochs=epochs, 
        weight_decay=wd, 
        report_to='wandb', 
        run_name=f'{model_name}/{exp_name}/fold-{i}', 
        save_strategy='epoch',
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        # max_grad_norm=1000,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    if resize:
        model.resize_token_embeddings(len(tokenizer))
    
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokenizer, data_collator=collator)

for i in range(5):
    wandb.init(
        project='lmsys-chatbot-arena',
        name=f'{model_name}/{exp_name}/fold-{i}',
        group=f'{model_name}/{exp_name}',
        save_code=True
    )
    dds = get_fold(i)

    trainer = get_trainer(dds)

    trainer.train();
    wandb.finish()
    
    del trainer
    torch.cuda.empty_cache()
    gc.collect();