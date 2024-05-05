from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
import torch.nn as nn
import torch.nn.functional as F

import polars as pl
import wandb
import os, gc
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

exp_name = 'sequence_lr_1e-6_no_drop'
bs = 1
lr = 1e-6
epochs = 1
wd = 0.01
resize = False

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'

df = pl.read_parquet('data/train.parquet')
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    responses = [
        f"{r[0]}{sep_token}{r[1]}{sep_token}{r[2]}" for i, r in enumerate(chat_list)
    ]

    return {'prompt': responses}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    lambda x: tokenizer(x['prompt'], max_length=1024, truncation=True, padding=True), 
    batched=False,
    num_proc=8,
    remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
)

def get_fold(fold_num):
    return DatasetDict({
        "train": tok_ds.filter(lambda x: x['fold'] != fold_num), 
        "test": tok_ds.filter(lambda x: x['fold'] == fold_num)
    })

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class Transformer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer.config.update({
            'hidden_dropout_prob': 0.,
            'attention_probs_dropout_prob': 0.
        })
        self.pooling = MeanPooling()
        self.head = nn.Linear(self.transformer.config.hidden_size, 3)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        token_type_ids = token_type_ids.squeeze(0)

        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state

        # mean pooling
        out = self.pooling(out, attention_mask)
        out = out.mean(0).unsqueeze(0)

        logits = self.head(out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

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
        ddp_find_unused_parameters=False,
        # max_grad_norm=1000,
    )
    
    model = Transformer(model_name)
    model.transformer.gradient_checkpointing_enable()
    
    return Trainer(
        model, 
        args, 
        train_dataset=dds['train'], 
        eval_dataset=dds['test'],
        tokenizer=tokenizer, 
        # data_collator=collator
    )

for i in range(4):
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