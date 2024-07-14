from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file

import torch
import torch.nn as nn
import torch.nn.functional as F

import polars as pl
import os, gc
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

exp_name = 'cross_encoder_add_lmsys_len_1800'

bs = 16
max_length = 1800

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'
df = pl.concat([
    pl.read_parquet('data/preference_collection.parquet').select([
        'prompt', 'response_a', 'response_b', pl.col('labels').cast(pl.Int32), pl.col('fold').cast(pl.Int32)]),
    pl.read_parquet('data/ultrafeedback.parquet').select([
        'prompt', 'response_a', 'response_b', pl.col('labels').cast(pl.Int32), pl.col('fold').cast(pl.Int32)]),
])
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    responses = [f"{r[0]}{sep_token}{r[1]}{sep_token}{r[2]}{sep_token}" for r in chat_list]

    return {
        'formatted_prompt': ''.join(responses)
    }

def tokenize(batch):
    return tokenizer(batch['formatted_prompt'], max_length=max_length, truncation=True, padding=False)

def calc_length(example):
    return {'prompt_length': len(example['input_ids'])}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=True,
    num_proc=8,
    remove_columns='formatted_prompt',
)
tok_ds = (
    tok_ds
    .map(calc_length, num_proc=8)
    .sort('prompt_length', reverse=True)
    .remove_columns('prompt_length')
).with_format('torch')

prompts = list(tok_ds['prompt'])
response_a = list(tok_ds['response_a'])
response_b = list(tok_ds['response_b'])

tok_ds = tok_ds.remove_columns(['prompt', 'response_a', 'response_b'])

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
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_config(config)
        self.pooling = MeanPooling()
        self.head = nn.Linear(self.transformer.config.hidden_size, 3)        

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state
        out = self.pooling(out, attention_mask)

        logits = self.head(out)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

def get_trainer(checkpoint_path):
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
    )
    
    model = Transformer(model_name)
    model.load_state_dict(load_file(f'{checkpoint_path}/model.safetensors'))

    return Trainer(
        model, 
        args, 
        train_dataset=tok_ds, 
        eval_dataset=tok_ds,
        tokenizer=tokenizer, 
        data_collator=collator
    )

all_preds = None
n_folds = 1
for i in range(n_folds):
    checkpoint_path = f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}'
    checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
    trainer = get_trainer(checkpoint_path)
 
    predictions = trainer.predict(tok_ds).predictions
    if all_preds is None:
        all_preds = predictions / n_folds
    else:
        all_preds += (predictions / n_folds)
    
    del trainer
    torch.cuda.empty_cache()
    gc.collect();

all_preds = torch.from_numpy(all_preds).softmax(-1)
all_preds = all_preds.numpy()
tok_ds = (
    tok_ds
    .add_column('winner_model_a_pred', list(all_preds[:, 0]))
    .add_column('winner_model_b_pred', list(all_preds[:, 1]))
    .add_column('winner_tie_pred', list(all_preds[:, 2]))
    .add_column('prompt', prompts)
    .add_column('response_a', response_a)
    .add_column('response_b', response_b)
)
print(tok_ds['winner_model_a_pred'][:5])
save_path = f'data/pseudo/{model_name.replace("/", "_")}-{exp_name}.parquet'
tok_ds.to_parquet(save_path)