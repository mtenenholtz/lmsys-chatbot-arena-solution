from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

import polars as pl

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

model_name = 'microsoft/deberta-v3-large'
bs = 16
max_length = 1024

df = pl.read_parquet('data/train.parquet')
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    responses = [f"{r[0]}{sep_token}{r[1]}{sep_token}{r[2]}{sep_token}" for r in chat_list]

    return {
        'prompt': ''.join(responses)
    }

def tokenize(batch):
    return tokenizer(batch['prompt'], max_length=max_length, truncation=True, padding=False)

def calc_length(example):
    return {'prompt_length': len(example['input_ids'])}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=True,
    num_proc=8,
    remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
)
tok_ds = (
    tok_ds
    .map(calc_length, num_proc=8)
    .sort('prompt_length', reverse=True)
    .remove_columns('prompt_length')
).with_format('torch')

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
        self.model = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooling()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
        return self.pooler(outputs, attention_mask)

model = Transformer(model_name)
collator = DataCollatorWithPadding(tokenizer)

args = TrainingArguments(
    'output',
    bf16=True,
    bf16_full_eval=True,
    tf32=True,
    per_device_train_batch_size=bs, 
    per_device_eval_batch_size=bs,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
    eval_accumulation_steps=1,
)

trainer = Trainer(
    model,
    args,
    train_dataset=tok_ds,
    eval_dataset=tok_ds,
    tokenizer=tokenizer,
    data_collator=collator
)

predictions = trainer.predict(tok_ds).predictions
embed_dset = Dataset.from_dict({"embeddings": predictions})
tok_ds = concatenate_datasets([tok_ds, embed_dset], axis=1)
print(tok_ds['embeddings'][:5])
tok_ds.to_parquet(f'data/embeddings/{model_name.replace("/", "_")}.parquet')