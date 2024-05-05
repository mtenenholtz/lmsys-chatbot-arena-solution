from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput

from trl import SFTTrainer 
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

exp_name = 'sft_len_1024'
bs = 12
lr = 1e-5
epochs = 1
wd = 0.01
resize = False

max_length = 1024

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = 'meta-llama/Meta-Llama-3-8B'

df = pl.read_parquet('data/train.parquet')

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token

prompts = df['prompt'].to_list()
responses_a = df['response_a'].to_list()
responses_b = df['response_b'].to_list()
folds = df['fold'].to_list()

dataset_a = {
    'prompt': prompts,
    'response': responses_a,
    'fold': folds
}
dataset_b = {
    'prompt': prompts,
    'response': responses_b,
    'fold': folds
}

dataset = concatenate_datasets([
    Dataset.from_dict(dataset_a),
    Dataset.from_dict(dataset_b)
])

def formatting_func(example):
    text = ''
    for prompt, response in zip(example['prompt'], example['response']):
        text += f'Human: {prompt}\nAssistant: {response}\n'
    
    return text

# def formatting_func(example):
#     texts = []

#     prompts = example['prompt']
#     responses = example['response']

#     for prompt, response in zip(prompts, responses):
#         text = ''
#         for p, r in zip(prompt, response):
#             text += f'Human: {p}\nAssistant: {r}\n'

#         texts.append(text)

#     return {'text': texts}

def get_fold(fold_num):
    return DatasetDict({
        "train": dataset.filter(lambda x: x['fold'] != fold_num), 
        "test": dataset.filter(lambda x: x['fold'] == fold_num)
    })

def get_trainer(dds):
    args = TrainingArguments(
        f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}',
        learning_rate=lr, warmup_steps=0, 
        # gradient_accumulation_steps=accum,
        lr_scheduler_type='cosine', 
        bf16=True,
        bf16_full_eval=True,
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
        run_name=f'{model_name}/{exp_name}/fold-{i}', 
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
        device_map={'': torch.cuda.current_device()}
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
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
    # train_dataset = dds['train'].map(formatting_func, batched=True)
    # eval_dataset = dds['test'].map(formatting_func, batched=True)

    return SFTTrainer(
        model, 
        args=args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, 
        max_seq_length=max_length,
        formatting_func=formatting_func,
        dataset_text_field='text',
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