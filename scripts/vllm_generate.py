import os
# os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import random
random.seed(34)

import polars as pl
import torch
import gc

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

models = [
    # 'HuggingFaceH4/zephyr-7b-beta',
    # 'lmsys/vicuna-13b-v1.3',
    # 'google/gemma-2-9b-it',
    'Qwen/Qwen2-7B-Instruct',
    'microsoft/Phi-3-small-8k-instruct',
    '01-ai/Yi-1.5-9B-Chat',
    'Nexusflow/Starling-LM-7B-beta',
    # 'modelscope/Yi-1.5-34B-Chat-AWQ',
    'Qwen/Qwen1.5-14B-Chat-AWQ',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]

def select_model(example):
    model = random.choice(models)
    return {'model': model if isinstance(model, str) else model[1]}

ds = load_dataset('lmsys/lmsys-chat-1m', split='train')
# ds = ds.filter(lambda x: any(s in x['model'] for s in ['gpt-4', 'gpt-3.5', 'claude']), num_proc=8)
ds = ds.shuffle(seed=42)
ds = ds.select(range(500_000))
ds = ds.map(select_model)
ds = ds.map(lambda x: {'prompt': [c['content'] for c in x['conversation'] if c['role'] == 'user']})
ds = ds.map(lambda x: {'turns': len(x['prompt'])})

all_ds = []
for model in models:
    print(f'Generating for model: {model}')

    if isinstance(model, tuple):
        llm = LLM(
            model=model[0],
            dtype='float16',
            quantization='awq',
            # tensor_parallel_size=2,
            seed=34,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model[1], trust_remote_code=True)
        model_name = model[1]
    elif 'awq' in model.lower():
        llm = LLM(
            model=model,
            dtype='float16',
            quantization='awq',
            seed=34,
            max_model_len=8192
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_name = model
    elif 'starling' in model.lower():
        llm = LLM(
            model=model,
            dtype='bfloat16',
            # tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            seed=34,
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_name = model
    elif 'mistral' in model.lower() or 'qwen' in model.lower():
        llm = LLM(
            model=model,
            dtype='bfloat16',
            # tensor_parallel_size=2,
            seed=34,
            max_model_len=30832
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_name = model
    elif 'gemma-2-9b' in model.lower():
        llm = LLM(
            model=model,
            dtype='bfloat16',
            # quantization='fp8',
            # tensor_parallel_size=2,
            seed=34,
            gpu_memory_utilization=0.9,
            max_model_len=2048
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_name = model
    elif 'phi-3' in model.lower():
        llm = LLM(
            model=model,
            dtype='bfloat16',
            # quantization='fp8',
            # tensor_parallel_size=2,
            seed=34,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model_name = model
    else:
        llm = LLM(
            model=model,
            dtype='bfloat16',
            # tensor_parallel_size=2,
            seed=34,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model_name = model
    
    assert hasattr(tokenizer, 'chat_template')
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)

    model_ds = ds.filter(lambda x: x['model'] == model_name)
    lengths = model_ds.unique('turns')
    for length in lengths:
        length_ds = model_ds.filter(lambda x: x['turns'] == length)
        prompts = length_ds['prompt']

        turn_chats = []
        for turn in range(length):
            turn_prompts = [[{'role': 'user', 'content': p[turn]}] for p in prompts]
            if turn == 0:
                turn_chats += turn_prompts
            else:
                for i, p in enumerate(turn_prompts):
                    c = turn_chats[i]
                    c += p
                    turn_chats[i] = c
            
            tokenized_prompts = [tokenizer.apply_chat_template(p, add_generation_prompt=True) for p in turn_chats]
            generations = llm.generate(prompt_token_ids=tokenized_prompts, sampling_params=sampling_params)
            for i, response in enumerate(generations):
                turn_chats[i] += [{'role': 'assistant', 'content': response.outputs[0].text.strip()}]

        length_ds = length_ds.add_column('generated_conversation', turn_chats)
        all_ds.append(length_ds)

    del llm
    torch.cuda.empty_cache()
    gc.collect();

def parse_responses(example):
    if random.random() < 0.5:
        return {
            'response_a': [r['content'] for r in example['generated_conversation'] if r['role'] == 'assistant'],
            'response_b': [r['content'] for r in example['conversation'] if r['role'] == 'assistant']
        }
    else:
        return {
            'response_a': [r['content'] for r in example['conversation'] if r['role'] == 'assistant'],
            'response_b': [r['content'] for r in example['generated_conversation'] if r['role'] == 'assistant']
        }

all_ds = concatenate_datasets(all_ds)
all_ds = all_ds.map(parse_responses)
all_ds = pl.from_pandas(all_ds.to_pandas())
all_ds.write_parquet('data/generated_500k.parquet')