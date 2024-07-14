from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder
from tokenizers import AddedToken

from models.sw_transformer import SlidingWindowTransformerModel

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import polars as pl
import os, gc
os.environ['WANDB_PROJECT'] = 'lmsys-chatbot-arena'
import wandb
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

exp_name = 'sliding_window_aux_loss_0.1'
bs = 4
accum = 2
lr = 5e-6
epochs = 1
wd = 0.01
resize = False

max_length = 1800

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'

# df = pl.read_parquet('data/train.parquet')
df = pl.concat([
    pl.read_parquet('data/train.parquet')
        .select([pl.col('id'), 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'labels', 'fold']),
    pl.read_parquet('data/lmsys-33k-deduplicated.parquet')
        .select([pl.col('id').cast(pl.Int64), 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'labels', 'fold'])
])
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
    
def map_model(example):
    return {
        'model_a_labels': model_name_map[example['model_a']],
        'model_b_labels': model_name_map[example['model_b']],
    }

def calc_length(example):
    return {'prompt_length': len(example['input_ids'])}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

model_names = list(set(df['model_a'].unique().to_list() + df['model_b'].unique().to_list()))
model_name_map = {n: i for i, n in enumerate(sorted(model_names))}

ds = ds.map(format_prompt, num_proc=8, batched=False)
ds = ds.map(map_model, num_proc=8, batched=False, remove_columns=['model_a', 'model_b'])
tok_ds = ds.map(
    tokenize, 
    batched=True,
    num_proc=8,
    remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold', 'model_a_labels', 'model_b_labels']],
)

def get_fold(fold_num):
    return DatasetDict({
        "train": tok_ds.filter(lambda x: x['fold'] != fold_num), 
        "test": (
            tok_ds
            .filter(lambda x: x['fold'] == fold_num)
            .map(calc_length, num_proc=8)
            .sort('prompt_length', reverse=True)
            .remove_columns('prompt_length')
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
        out = self.transformer_head(out, attention_mask).last_hidden_state
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
        learning_rate=lr, warmup_ratio=0., 
        gradient_accumulation_steps=accum,
        lr_scheduler_type='cosine', 
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
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
        run_name=f'{model_name}/{exp_name}/fold-{i}'.replace('/', '_').replace('/mnt/one/kaggle/lmsys-chatbot-arena/', ''), 
        save_strategy='epoch',
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        # gradient_checkpointing=True,
        # label_smoothing_factor=0.1
        # max_grad_norm=1000,
    )
    
    # model = Transformer(model_name)
    # model.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = SlidingWindowTransformerModel(model_name, window_size=max_length)
    model.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokenizer, data_collator=collator)

for i in range(4):
    # wandb.init(
    #     project='lmsys-chatbot-arena',
    #     name=f'{model_name}/{exp_name}/fold-{i}',
    #     group=f'{model_name}/{exp_name}',
    #     save_code=True
    # )
    dds = get_fold(i)

    trainer = get_trainer(dds)
 
    trainer.train();
    wandb.finish()
    
    del trainer
    torch.cuda.empty_cache()
    gc.collect();