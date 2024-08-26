from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel

from models.llm_model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import polars as pl
import os, gc, sys, argparse, yaml
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

from types import SimpleNamespace

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
print(arguments)

exp_name = arguments.exp_name
bs = arguments.validation.batch_size
accum = arguments.validation.accum
lr = arguments.lr
epochs = arguments.epochs
wd = arguments.weight_decay
resize = False
do_tta = arguments.tta

max_length = arguments.validation.max_length

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = arguments.model_name

df = pl.read_parquet('data/train.parquet')
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    if not do_tta:
        # responses = [f"Prompt: {r[0]}\nResponse A: {r[1]}\nResponse B: {r[2]}\n" for r in chat_list]
        responses = [f"<PROMPT>{r[0]}</PROMPT><RESPONSE A>{r[1]}</RESPONSE A><RESPONSE B>{r[2]}</RESPONSE B>" for r in chat_list]
    else:
        # responses = [f"Prompt: {r[0]}\nResponse A: {r[2]}\nResponse B: {r[1]}\n" for r in chat_list]
        responses = [f"<PROMPT>{r[0]}</PROMPT><RESPONSE A>{r[2]}</RESPONSE A><RESPONSE B>{r[1]}</RESPONSE B>" for r in chat_list]

    return {
        'formatted_prompt': ''.join(responses)
    }

def tokenize(batch):
    output = tokenizer(
        batch['formatted_prompt'], 
        max_length=max_length, 
        truncation=True, 
        padding=False
    )

    if len(output['input_ids']) > max_length:
        input_ids = [tokenizer.bos_token_id] + output['input_ids'][-(max_length-1):]
        attention_mask = [1] * len(input_ids)
    else:
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    # return tokenizer(
    #     batch['formatted_prompt'], 
    #     max_length=max_length, 
    #     truncation=True, 
    #     padding=False
    # )

# def format_and_tokenize(row):
#     chat_list = zip(row['prompt'], row['response_a'], row['response_b'])

#     indiv_len = int(max_length // 3 // len(row['prompt']))
#     response_input_ids = []
#     for r in chat_list:
#         prompt_tok = tokenizer('Prompt: ' + str(r[0]) + '\n', max_length=indiv_len, truncation=True, padding=False, add_special_tokens=False)
#         response_a_tok = tokenizer('Response A: ' + str(r[1]) + '\n', max_length=indiv_len, truncation=True, padding=False, add_special_tokens=False)
#         response_b_tok = tokenizer('Response B: ' + str(r[2]) + '\n', max_length=indiv_len, truncation=True, padding=False, add_special_tokens=False)
        
#         turn_input_ids = prompt_tok['input_ids'] + response_a_tok['input_ids'] + response_b_tok['input_ids']
        
#         response_input_ids += turn_input_ids
        
#     response_input_ids = [128000] + response_input_ids

#     return {
#         'input_ids': response_input_ids,
#         'attention_mask': [1] * len(response_input_ids),
#         'formatted_prompt': tokenizer.decode(response_input_ids),
#     }

def calc_length(example):
    return {'prompt_length': len(example['input_ids'])}

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
sep_token = tokenizer.sep_token
if 'qwen' in model_name.lower():
    tokenizer.padding_side = 'left'
elif 'mistral-nemo' in model_name.lower():
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.pad_token = tokenizer.eos_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=False,
    num_proc=8,
)
# tok_ds = ds.map(
#     format_and_tokenize,
#     batched=False,
#     num_proc=8,
# )

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
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    ckpt_dir = os.listdir(f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}/')[0]
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name,
    #     quantization_config=quant_config,
    #     num_labels=3,
    #     pad_token_id=tokenizer.pad_token_id,
    #     attn_implementation='flash_attention_2',
    # )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = Model(config, model_name, quant_config=quant_config, pad_token_id=tokenizer.pad_token_id, training=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.attn_logit_softcapping = None
    model = PeftModel.from_pretrained(
        model, 
        f'{ckpt_base_dir}/{model_name}-{exp_name}-fold-{i}/{ckpt_dir}'
    )

    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokenizer, data_collator=collator)

dfs = []
for i in range(1):
    dds = get_fold(i)
    
    trainer = get_trainer(dds)
    fold_logits = trainer.predict(dds['test']).predictions
    if do_tta:
        fold_logits = fold_logits[:, [1, 0, 2]]
    fold_probs = np.array(torch.from_numpy(fold_logits).softmax(-1))
    # if do_tta:
    #     fold_probs = fold_probs[:, [1, 0, 2]]
    ids = np.array(dds['test']['id'])[:, None]
    lengths = np.array(dds['test']['prompt_length'])[:, None]

    fold_output = np.concatenate((ids, lengths, fold_logits, fold_probs), axis=1)
    fold_df = pl.from_numpy(
        fold_output, 
        ['id', 'prompt_length', 'logits_model_a', 'logits_model_b', 'logits_tie', 'prob_model_a', 'prob_model_b', 'prob_tie']
    )
    fold_df = fold_df.with_columns(formatted_prompt=pl.Series(list(dds['test']['formatted_prompt'])))
    dfs.append(fold_df)
 
    del trainer
    torch.cuda.empty_cache()
    gc.collect();

df = pl.concat(dfs)
if do_tta:
    df.write_parquet(f'data/preds/{model_name.replace("/", "_")}-{exp_name}_tta.parquet')
else:
    df.write_parquet(f'data/preds/{model_name.replace("/", "_")}-{exp_name}.parquet')

