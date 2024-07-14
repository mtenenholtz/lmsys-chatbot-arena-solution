import ctranslate2 as ct2
from transformers import AutoTokenizer

from textwrap import dedent
from tqdm import tqdm

import polars as pl
from datasets import Dataset

import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

model = ct2.Generator('/mnt/one/ct2_models/meta-llama/Meta-Llama-3-8B-Instruct', device='cuda', device_index=[0, 1])
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

df = pl.read_parquet('data/train.parquet')
ds = Dataset.from_pandas(df.to_pandas())

def format_prompt(row):
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    messages = [
        {
            'role': 'system', 
            'content': dedent("""
                You are a human preference evaluator. 
            """.replace('\n', ''))
         }
    ]
    for prompt, response_a, response_b in chat_list:
        messages.append({'role': 'user', 'content': prompt})
        messages.append({'role': 'assistant', 'content': f'Responder A: {response_a}\nResponder B: {response_b}'})

    return {
        'prompt': messages,
    }

def tokenize(example):
    ids = tokenizer.apply_chat_template(
        example['prompt'],
        add_generation_prompt=True,
        max_length=4096,
    )
    
    evaluation_messages = [{
        'role': 'user', 
        'content': dedent("""
            Given the user prompt and the two competing responders (A and B) with potentially truncated results, 
            first, explain the content of what the user was looking for, and any specific asks that the user had along the way. 
            Then, list the pros and cons of each responder\'s responses to those requests.
            Finally, make a judgement which responder provided a better set of responses, or if it was a tie.
            Only refer to the responders as "Responder A" and "Responder B".
        """)
    }]
    evaluation_ids = tokenizer.apply_chat_template(
        evaluation_messages,
        add_generation_prompt=True,
        max_length=7000,
    )

    ids = ids + evaluation_ids
    return {'prompt_tokens': tokenizer.convert_ids_to_tokens(ids)}

def calc_length(example):
    return {'prompt_length': len(example['prompt_tokens'])}

ds = ds.map(format_prompt, num_proc=8, batched=False)
tok_ds = ds.map(
    tokenize, 
    batched=False,
    num_proc=12,
)
tok_ds = tok_ds.map(calc_length, num_proc=12)
tok_ds = tok_ds.sort('prompt_length')

eos_token_id = tokenizer.eos_token_id
eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
end_tokens = [eos_token_id, eot_token_id]

dataset = tok_ds.flatten()
batches = [dataset[i:i + 10] for i in range(0, len(dataset), 10)]

batch_size = 10

for batch in tqdm(batches):
    prompt_tokens = batch['prompt_tokens']
    results = model.generate_batch(
        prompt_tokens,
        max_batch_size=10,
        # sampling_temperature=0.8,
        # sampling_topk=20,
        max_length=8192,
        sampling_temperature=0.6,
        sampling_topp=0.9,
        end_token=end_tokens,
        include_prompt_in_result=False,
    )

idx = 4
print(tokenizer.decode(results[idx].sequences_ids[0]))
print('')
label_map = {
    0: 'Model A Wins',
    1: 'Model B Wins',
    2: 'Tie'
}
print(f'Answer: {label_map[batches[-1]["labels"][idx]]}')
print(f'Model A: {batches[-1]["model_a"][idx]}')
print(f'Model B: {batches[-1]["model_b"][idx]}')

# print(tokenizer.decode(results[0].sequences_ids[0]))
# print(list(results))