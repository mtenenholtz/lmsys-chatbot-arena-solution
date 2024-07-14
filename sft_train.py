from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from accelerate import Accelerator

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from unsloth import FastLanguageModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import polars as pl
import os, gc
os.environ['WANDB_PROJECT'] = 'lmsys-chatbot-arena-lm'
import wandb
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

exp_name = 'sft_len_1800_surround_model_prefix'
bs = 8
accum = 1
lr = 5e-5
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
        .select([pl.col('id').cast(pl.Int64), 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'labels', 'fold']),
    # pl.read_parquet('data/mt_bench_human_judgments.parquet')
    #     .select([pl.col('id').cast(pl.Int64), 'model_a', 'model_b', 'prompt', 'response_a', 'response_b', 'labels', 'fold'])
])

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
sep_token = tokenizer.sep_token
tokenizer.pad_token = tokenizer.eos_token
if 'qwen' in model_name.lower():
    tokenizer.padding_side = 'left'

prompts = df['prompt'].to_list()
responses_a = df['response_a'].to_list()
responses_b = df['response_b'].to_list()
model_a = df['model_a'].to_list()
model_b = df['model_b'].to_list()
folds = df['fold'].to_list()

dataset = {
    'prompt': prompts,
    'response_a': responses_a,
    'response_b': responses_b,
    'model_a': model_a,
    'model_b': model_b,
    'fold': folds
}

dataset = Dataset.from_dict(dataset)

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    responses = [f"<PROMPT>{r[0]}</PROMPT><RESPONSE A>{r[1]}</RESPONSE A><RESPONSE B>{r[2]}</RESPONSE B>" for r in chat_list]
    
    chat = ''.join(responses)

    prefix = f'Model A: {row["model_a"]}\nModel B: {row["model_b"]}\n'

    return prefix + chat

def calc_length(example):
    return {'prompt_length': len(example['prompt']) + len(example['response_a']) + len(example['response_b'])}

def get_fold(fold_num):
    return DatasetDict({
        "train": dataset.filter(lambda x: x['fold'] != fold_num), 
        "test": (
            dataset
            .filter(lambda x: x['fold'] == fold_num)
            .map(calc_length, num_proc=8)
            .sort('prompt_length', reverse=True)
            .remove_columns('prompt_length')
        )
    })

def get_trainer(dds):
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
        # group_by_length=True,
        num_train_epochs=epochs, 
        weight_decay=wd, 
        report_to='wandb', 
        run_name=f'{model_name}/{exp_name}/fold-{i}'.replace('/', '_'), 
        save_strategy='epoch',
        save_total_limit=1,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        # gradient_checkpointing=True,
        # max_grad_norm=1000,
    )
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True,
        device_map={'': Accelerator().process_index}
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 
            'gate_proj', 'up_proj', 'down_proj',
            # 'embed_tokens', 'lm_head', 
        ],
        # modules_to_save=['lm_head'],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = max_length,
    )

    train_dataset = dds['train']
    eval_dataset = dds['test']
    
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=format_prompt,
        seq_length=max_length,
    )
    eval_dataset = ConstantLengthDataset(
        tokenizer,
        eval_dataset,
        formatting_func=format_prompt,
        seq_length=max_length,
        shuffle=False
    )

    return SFTTrainer(
        model, 
        args=args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        max_seq_length=max_length,
        tokenizer=tokenizer, 
        packing=True
    )

for i in range(4):
    dds = get_fold(i)

    trainer = get_trainer(dds)

    trainer.train();
    wandb.finish()
    
    del trainer
    torch.cuda.empty_cache()
    gc.collect();