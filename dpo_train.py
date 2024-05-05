from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput

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

exp_name = 'dpo'
bs = 16
lr = 1e-5
epochs = 1
wd = 0.01
resize = False

max_length = 1024

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'

df = pl.read_parquet('data/train.parquet')
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    a_chat = zip(row['prompt'], row['response_a'])
    b_chat = zip(row['prompt'], row['response_b'])

    a_responses = [f"Human: {r[0]}\nAssistant: {r[1]}\n" for r in a_chat]
    b_responses = [f"Human: {r[0]}\nAssistant: {r[1]}\n" for r in b_chat]

    return {
        'prompt_a': ''.join(a_responses),
        'prompt_b': ''.join(b_responses),
    }

def tokenize(batch):
    tok_a = tokenizer(batch['prompt_a'], max_length=max_length, truncation=True)
    tok_a = {k: v for k, v in tok_a.items()}

    tok_b = tokenizer(batch['prompt_b'], max_length=max_length, truncation=True)
    tok_b = {f'{k}_b': v for k, v in tok_b.items()}

    return {**tok_a, **tok_b}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=True,
    num_proc=8,
    remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
)

def get_fold(fold_num):
    return DatasetDict({
        "train": tok_ds.filter(lambda x: x['fold'] != fold_num), 
        "test": tok_ds.filter(lambda x: x['fold'] == fold_num)
    })

class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_a = [{
            'input_ids': e['input_ids'], 
            'attention_mask': e['attention_mask'], 
            'token_type_ids': e['token_type_ids']
        } for e in batch]
        batch_b = [{
            'input_ids': e['input_ids_b'], 
            'attention_mask': e['attention_mask_b'], 
            'token_type_ids': e['token_type_ids_b']
        } for e in batch]
        
        batch_a_padded = self.tokenizer.pad(
            batch_a,
            padding=True,
            return_tensors="pt"
        )
        batch_b_padded = self.tokenizer.pad(
            batch_b,
            padding=True,
            return_tensors="pt"
        )
        
        labels = torch.tensor([example['labels'] for example in batch])
        
        return {
            'input_ids': batch_a_padded['input_ids'],
            'attention_mask': batch_a_padded['attention_mask'],
            'token_type_ids': batch_a_padded['token_type_ids'],
            'input_ids_b': batch_b_padded['input_ids'],
            'attention_mask_b': batch_b_padded['attention_mask'],
            'token_type_ids_b': batch_b_padded['token_type_ids'],
            'labels': labels
        }


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
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name)

    def get_batch_logps(self, logits, labels):
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != tokenizer.pad_token_id

        labels[labels == tokenizer.pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1)
    
    def forward(
            self, 
            input_ids, attention_mask, token_type_ids, 
            input_ids_b, attention_mask_b, token_type_ids_b, 
            labels=None
        ):
        logits_a = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits

        logits_b = self.transformer(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            token_type_ids=token_type_ids_b
        ).logits

        logps_a = self.get_batch_logps(logits_a, input_ids)
        logps_b = self.get_batch_logps(logits_b, input_ids_b)

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
        # gradient_checkpointing=True,
        # max_grad_norm=1000,
    )
    
    model = Transformer(model_name)
    model.transformer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

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