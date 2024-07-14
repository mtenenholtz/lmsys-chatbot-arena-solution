from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model

from models.biencoder import BiEncoder

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

exp_name = 'llm_biencoder_qkvo'
bs = 2
accum = 4
lr = 1e-4
epochs = 1
wd = 0.01
resize = False

max_length = 1800

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

df = pl.concat([
    pl.read_parquet('data/train.parquet')
        .select([pl.col('id'), 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'labels', 'fold']),
    pl.read_parquet('data/lmsys-33k-deduplicated.parquet')
        .select([pl.col('id').cast(pl.Int64), 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'labels', 'fold'])
])
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])

    response_a = [f"Prompt: {r[0]}\nResponse: {r[1]}\n" for r in chat_list]
    response_b = [f"Prompt: {r[0]}\nResponse: {r[2]}\n" for r in chat_list]

    return {
        'prompt_a': ''.join(response_a),
        'prompt_b': ''.join(response_b),
    }

def tokenize(batch):
    outputs_a = tokenizer(
        batch['prompt_a'], 
        max_length=max_length, 
        truncation=True, 
        padding=False
    )
    outputs_b = tokenizer(
        batch['prompt_b'], 
        max_length=max_length, 
        truncation=True, 
        padding=False
    )
    outputs_b = {f'{k}_b': v for k, v in outputs_b.items()}

    return {**outputs_a, **outputs_b}

def calc_length(example):
    return {'prompt_length': max(len(example['input_ids']), len(example['input_ids_b']))}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token
tokenizer.pad_token = tokenizer.eos_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=False,
    num_proc=8,
    remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
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

def get_trainer(dds):
    collator = DataCollatorWithPadding(tokenizer)
    args = TrainingArguments(
        f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}',
        learning_rate=lr, warmup_steps=10, 
        gradient_accumulation_steps=accum,
        lr_scheduler_type='cosine', 
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        optim='paged_adamw_8bit',
        evaluation_strategy="epoch", 
        logging_steps=1,
        per_device_train_batch_size=bs, 
        per_device_eval_batch_size=bs,
        greater_is_better=False, 
        group_by_length=True,
        num_train_epochs=epochs, 
        weight_decay=wd, 
        report_to='wandb', 
        run_name=f'{model_name}/{exp_name}/fold-{i}'.replace('/', '_'), 
        save_strategy='epoch',
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}
        # deepspeed='deepspeed/zero2.json',
        # max_grad_norm=1000,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    config = AutoConfig.from_pretrained(model_name)
    config.pad_token_id = tokenizer.pad_token_id
    model = BiEncoder(config, model_name, quant_config)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.,
        bias='none',
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        modules_to_save=['head']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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