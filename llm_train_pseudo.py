from datasets import Dataset, DatasetDict
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, 
    LlamaForCausalLM, LlamaPreTrainedModel,
    TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model

from models.llm_model import Model
from models.sw_transformer import SlidingWindowLLM

import torch
import torch.nn as nn
import torch.nn.functional as F

import polars as pl
import os, gc, sys
os.environ['WANDB_PROJECT'] = 'lmsys-chatbot-arena'
import yaml
import wandb
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)
from types import SimpleNamespace
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
args = yaml.safe_load(open(parser_args.config).read())
for k, v in args.items():
    if type(v) == dict:
        args[k] = SimpleNamespace(**v)
arguments = SimpleNamespace(**args)
print(args)

exp_name = arguments.exp_name
bs = arguments.training.batch_size
accum = arguments.training.accum
lr = arguments.lr
epochs = arguments.epochs
wd = arguments.weight_decay
resize = False

max_length = arguments.training.max_length

ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
model_name = arguments.model_name

df = pl.concat([
    pl.read_parquet('data/train.parquet')
        .select([
            pl.col('id'), 'prompt', 'response_a', 'response_b', 
            'winner_model_a', 'winner_model_b', 'winner_tie', 
            'fold'
        ]),
    pl.read_parquet('data/lmsys-33k-deduplicated.parquet')
        .select([
            pl.col('id').cast(pl.Int64), 'prompt', 'response_a', 'response_b', 
            'winner_model_a', 'winner_model_b', 'winner_tie', 
            'fold'
        ])
])
df = df.with_columns(
    winner_model_a=pl.col('winner_model_a').cast(pl.Float32),
    winner_model_b=pl.col('winner_model_b').cast(pl.Float32),
    winner_tie=pl.col('winner_tie').cast(pl.Float32),
)

ext_df = pl.concat([
    # pl.read_parquet('data/preference_collection.parquet').select([
    #     'prompt', 'response_a', 'response_b', pl.col('fold').cast(pl.Int32)]),
    # pl.read_parquet('data/ultrafeedback.parquet').select([
    #     'prompt', 'response_a', 'response_b', pl.col('fold').cast(pl.Int32)]),
    # pl.read_parquet('data/Capybara-Preferences.parquet').select([
    #     'prompt', 'response_a', 'response_b', pl.col('fold').cast(pl.Int32)]),
    pl.read_parquet('data/orpo-dpo-mix-40k.parquet').select([
        'prompt', 'response_a', 'response_b', pl.col('fold').cast(pl.Int32)]),
    pl.read_parquet('data/generated.parquet').select([
        'prompt', 'response_a', 'response_b', pl.lit(-1).alias('fold')]),
]).with_row_index(name='id').with_columns(id=(pl.col('id') + 1) * -1)
# ext_df = ext_df.filter(pl.col('source') == 'preferences')
ext_df = ext_df.drop('source')
pseudo_labels = pl.read_parquet('data/pseudo/opt_pseudos_fold_0_sampled.parquet')
ext_df = (
    ext_df
    .join(pseudo_labels, on='id')
    .with_columns(
        id=pl.col('id').cast(pl.Int64),
        winner_model_a=pl.col('winner_model_a_pred').cast(pl.Float32),
        winner_model_b=pl.col('winner_model_b_pred').cast(pl.Float32),
        winner_tie=pl.col('winner_tie_pred').cast(pl.Float32),
    )
    .select(df.columns)
    .drop(['winner_model_a_pred', 'winner_model_b_pred', 'winner_tie_pred'])
)
df = (
    df
    .with_columns(
        winner_model_a=pl.col('winner_model_a').cast(pl.Float32),
        winner_model_b=pl.col('winner_model_b').cast(pl.Float32),
        winner_tie=pl.col('winner_tie').cast(pl.Float32),
    )
)
df = (
    pl.concat([df, ext_df])
    .with_columns(labels=pl.concat_list([pl.col('winner_model_a'), pl.col('winner_model_b'), pl.col('winner_tie')]))
    .drop(['winner_model_a', 'winner_model_b', 'winner_tie'])
)
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    responses = [f"<PROMPT>{r[0]}</PROMPT><RESPONSE A>{r[1]}</RESPONSE A><RESPONSE B>{r[2]}</RESPONSE B>" for r in chat_list]

    return {
        'prompt': ''.join(responses)
    }

# def format_prompt(row):
#     chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
#     responses = [f"{prompt_start_tok}{r[0]}{prompt_end_tok}{response_a_start_tok}{r[1]}{response_a_end_tok}{response_b_start_tok}{r[2]}{response_b_end_tok}" for r in chat_list]

#     return {
#         'prompt': ''.join(responses)
#     }

def tokenize(batch):
    return tokenizer(
        batch['prompt'], 
        max_length=max_length, 
        truncation=True, 
        padding=False
    )

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
#     }

# def format_and_tokenize(row):
#     chat_list = zip(row['prompt'], row['response_a'], row['response_b'])

#     response_a_input_ids = []
#     for r in chat_list:
#         prompt_tok = tokenizer('Prompt: ' + str(r[0]) + '\n', add_special_tokens=False)
#         response_a_tok = tokenizer('Response A: ' + str(r[1]) + '\n', add_special_tokens=False)
        
#         turn_input_ids = prompt_tok['input_ids'] + response_a_tok['input_ids']
        
#         response_a_input_ids += turn_input_ids

#     chat_list = zip(row['prompt'], row['response_b'])
    
#     response_b_input_ids = []
#     for r in chat_list:
#         prompt_tok = tokenizer('Prompt: ' + str(r[0]) + '\n', add_special_tokens=False)
#         response_b_tok = tokenizer('Response B: ' + str(r[1]) + '\n', add_special_tokens=False)
        
#         turn_input_ids = prompt_tok['input_ids'] + response_b_tok['input_ids']
        
#         response_b_input_ids += turn_input_ids
        
#     response_input_ids = [tokenizer.bos_token_id] + response_a_input_ids[:(max_length - 1) // 2] + [tokenizer.eos_token_id] + response_b_input_ids[:(max_length - 1) // 2]

#     return {
#         'input_ids': response_input_ids,
#         'attention_mask': [1] * len(response_input_ids),
#     }

def calc_length(example):
    return {'prompt_length': len(example['input_ids'])}

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
sep_token = tokenizer.sep_token
tokenizer.pad_token = tokenizer.eos_token

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=False,
    num_proc=8,
    remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
)
# tok_ds = ds.map(
#     format_and_tokenize,
#     batched=False,
#     num_proc=8,
#     remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
# )

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

class CustomTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and ('score' not in n and 'embed_tokens' not in n) and p.requires_grad)
                    ],
                    "learning_rate": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and ('score' in n or 'embed_tokens' in n) and p.requires_grad)
                    ],
                    "learning_rate": 1e-5,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        manager.register_module_override(module, "weight", {"optim_bits": 32})

        return self.optimizer


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
        # deepspeed='deepspeed/zero2.json',
        max_grad_norm=1000.,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name,
    #     num_labels=3,
    #     quantization_config=quant_config,
    #     pad_token_id=tokenizer.pad_token_id,
    #     attn_implementation='flash_attention_2',
    #     trust_remote_code=True,
    # )
    config = AutoConfig.from_pretrained(model_name)
    model = Model(config, model_name, quant_config=quant_config, pad_token_id=tokenizer.pad_token_id)
    model.config.pad_token_id = tokenizer.pad_token_id
    # model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=arguments.lora_r,
        lora_alpha=arguments.lora_alpha,
        target_modules=arguments.target_modules,
        lora_dropout=0. if not hasattr(arguments, 'lora_dropout') else arguments.lora_dropout,
        bias='none',
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        modules_to_save=[
            'score', 'lstm',
        ],
    )
    print(lora_config.modules_to_save)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return CustomTrainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokenizer, data_collator=collator)

for i in range(1):
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