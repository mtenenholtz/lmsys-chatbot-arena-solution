from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput

from models.sw_transformer import SlidingWindowTransformerModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file

import numpy as np

import polars as pl
import gc, os
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

exp_name = 'sliding_window_1024_mod_prompt'
bs = 8
lr = 5e-6
epochs = 1
wd = 0.01
resize = False

max_length = 1800

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'

df = pl.read_parquet('data/train.parquet')
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    responses = [f"Prompt: {r[0]}\nResponse A: {r[1]}\nResponse B: {r[2]}{sep_token}" for r in chat_list]

    return {
        'prompt': ''.join(responses)
    }

def tokenize(batch):
    return tokenizer(
        batch['prompt'], 
        # max_length=max_length, 
        # truncation=True, 
        # padding=False
    )

def calc_length(example):
    return {'prompt_length': len(example['input_ids'])}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=True,
    num_proc=8,
)

def get_fold(fold_num):
    return DatasetDict({
        "train": tok_ds.filter(lambda x: x['fold'] != fold_num), 
        "test": (
            tok_ds
            .filter(lambda x: x['fold'] == fold_num)
            .map(calc_length, num_proc=8)
            .sort('prompt_length', reverse=True)
        )
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
            'attention_probs_dropout_prob': 0.,
            'max_position_embeddings': 1024
        })
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

def get_trainer(dds):
    collator = DataCollatorWithPadding(tokenizer)
    args = TrainingArguments(
        f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}',
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        per_device_train_batch_size=bs, 
        per_device_eval_batch_size=bs,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )
    
    # model = Transformer(model_name)
    model = SlidingWindowTransformerModel(model_name, window_size=max_length)
    ckpt_dir = os.listdir(f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}/')[0]
    weights = load_file(f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}/{ckpt_dir}/model.safetensors')
    
    model.load_state_dict(weights)
    model = model.eval()
    
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokenizer, data_collator=collator)

dfs = []
for i in range(4):
    dds = get_fold(i)
    
    trainer = get_trainer(dds)
    fold_logits = trainer.predict(dds['test']).predictions
    fold_probs = np.array(torch.from_numpy(fold_logits).softmax(-1))
    ids = np.array(dds['test']['id'])[:, None]
    lengths = np.array(dds['test']['prompt_length'])[:, None]

    fold_output = np.concatenate((ids, lengths, fold_logits, fold_probs), axis=1)
    fold_df = pl.from_numpy(
        fold_output, 
        ['id', 'prompt_length', 'logits_model_a', 'logits_model_b', 'logits_tie', 'prob_model_a', 'prob_model_b', 'prob_tie']
    )
    fold_df = fold_df.with_columns(formatted_prompt=pl.Series(list(dds['test']['prompt'])))
    dfs.append(fold_df)
 
    del trainer
    torch.cuda.empty_cache()
    gc.collect();

df = pl.concat(dfs)
df.write_parquet(f'data/preds/{model_name.replace("/", "_")}-{exp_name}.parquet')