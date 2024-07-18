from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel

from models.llm_model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F

import polars as pl
import os, gc, sys, yaml
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)
from types import SimpleNamespace
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("--tta", action='store_true', help="whether to do test-time augmentation")
parser_args, _ = parser.parse_known_args(sys.argv)
args = yaml.safe_load(open(parser_args.config).read())
args['tta'] = parser_args.tta
for k, v in args.items():
    if type(v) == dict:
        args[k] = SimpleNamespace(**v)
arguments = SimpleNamespace(**args)
print(args)

exp_name = arguments.exp_name

bs = arguments.validation.batch_size
max_length = arguments.validation.max_length
do_tta = arguments.tta

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = arguments.model_name
df = pl.concat([
    pl.read_parquet('data/train.parquet')
        .select([pl.col('id'), 'prompt', 'response_a', 'response_b', 'labels', 'fold']),
    pl.read_parquet('data/lmsys-33k-deduplicated.parquet')
        .select([pl.col('id').cast(pl.Int64), 'prompt', 'response_a', 'response_b', 'labels', 'fold'])
])
ext_df = pl.concat([
    # pl.read_parquet('data/preference_collection.parquet').select([
    #     'prompt', 'response_a', 'response_b', pl.col('labels').cast(pl.Int32), pl.col('fold').cast(pl.Int32)]),
    # pl.read_parquet('data/ultrafeedback.parquet').select([
    #     'prompt', 'response_a', 'response_b', pl.lit(0).cast(pl.Int32).alias('labels'), pl.col('fold').cast(pl.Int32)]),
    # pl.read_parquet('data/Capybara-Preferences.parquet').select([
    #     'prompt', 'response_a', 'response_b', pl.lit(0).cast(pl.Int32).alias('labels'), pl.col('fold').cast(pl.Int32)]),
    # pl.read_parquet('data/distilabel-intel-orca-dpo-pairs.parquet').select([
    #     'prompt', 'response_a', 'response_b', pl.lit(0).cast(pl.Int32).alias('labels'), pl.col('fold').cast(pl.Int32)]),
    pl.read_parquet('data/orpo-dpo-mix-40k.parquet').select([
        'prompt', 'response_a', 'response_b', pl.lit(0).cast(pl.Int32).alias('labels'), pl.col('fold').cast(pl.Int32)]),
    pl.read_parquet('data/generated.parquet').select([
        'prompt', 'response_a', 'response_b', pl.lit(0).cast(pl.Int32).alias('labels'), pl.lit(-1).alias('fold')]),
]).with_row_index(name='id').with_columns(id=(pl.col('id') + 1) * -1)
# df = pl.concat([df, ext_df])
df = ext_df
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    if not do_tta:
        responses = [f"<PROMPT>{r[0]}</PROMPT><RESPONSE A>{r[1]}</RESPONSE A><RESPONSE B>{r[2]}</RESPONSE B>" for r in chat_list]
    else:
        responses = [f"<PROMPT>{r[0]}</PROMPT><RESPONSE A>{r[2]}</RESPONSE A><RESPONSE B>{r[1]}</RESPONSE B>" for r in chat_list]

    return {
        'formatted_prompt': ''.join(responses)
    }

def tokenize(batch):
    return tokenizer(batch['formatted_prompt'], max_length=max_length, truncation=True, padding=False)

def calc_length(example):
    return {'prompt_length': len(example['input_ids'])}

tokenizer = AutoTokenizer.from_pretrained(model_name)
sep_token = tokenizer.sep_token
tokenizer.pad_token = tokenizer.eos_token

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
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    ckpt_dir = os.listdir(f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}/')[0]
    
    config = AutoConfig.from_pretrained(model_name)
    model = Model(config, model_name, quant_config=quant_config, pad_token_id=tokenizer.pad_token_id, training=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model, 
        f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}/{ckpt_dir}'
    )

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
    if do_tta:
        predictions = predictions[:, [1, 0, 2]]
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

    if not do_tta:
        save_path = f'data/pseudo/{model_name.replace("/", "_")}-{exp_name}_fold_{i}.parquet'
    else:
        save_path = f'data/pseudo/{model_name.replace("/", "_")}-{exp_name}_tta_fold_{i}.parquet'
    tok_ds.to_parquet(save_path)