import ctranslate2
from transformers import AutoTokenizer
import random
random.seed(34)

import polars as pl

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

models = [
    # 'HuggingFaceH4/zephyr-7b-beta',
    # 'lmsys/vicuna-13b-v1.3',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3'
]

def stream_tokens(step_results, verbose=False):
    output_ids = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and output_ids:
            word = tokenizer.decode(output_ids)
            print(word, end=" ", flush=True)
            output_ids = []

        output_ids.append(step_result.token_id)

    if output_ids:
        word = tokenizer.decode(output_ids)
        print(word)

ds = load_dataset('lmsys/lmsys-chat-1m', split='train')
ds = ds.filter(lambda x: any(s in x['model'] for s in ['gpt-4', 'gpt-3.5', 'claude']), num_proc=8)
ds = ds.shuffle(seed=42)
ds = ds.select(range(4000))
ds = ds.map(lambda x: {'model': random.choice(models)})
ds = ds.map(lambda x: {'prompts': [c['content'] for c in x['conversation'] if c['role'] == 'user']})
ds = ds.map(lambda x: {'turns': len(x['prompts'])})

# all_ds = []
# for model in models:
#     model_ds = ds.filter(lambda x: x['model'] == model)
#     unique_lengths = model_ds.unique('turns')
#     generator = ctranslate2.Generator('/mnt/one/ct2_models/' + model, device='cuda')
#     tokenizer = AutoTokenizer.from_pretrained(model)

#     generations = []
#     for example in tqdm(list(model_ds['prompts'])):
#         prompt = []
#         prompt_generations = []
#         for message in example:
#             prompt.append(
#                 {"role": "user", "content": message}
#             )
#             chat_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
#             prompt_tokens = tokenizer.convert_ids_to_tokens(chat_ids)

#             step_results = generator.generate_batch(
#                 [prompt_tokens],
#                 max_length=512,
#                 sampling_topk=10,
#                 sampling_temperature=0.7,
#             )
#             # stream_tokens(step_results)
#             # print(list(step_results))
#             result = tokenizer.decode(tokenizer.convert_tokens_to_ids(step_results[0].sequences[0]))
#             # print(result)
#             prompt.append({
#                 "role": "assistant",
#                 "content": result
#             })
#             prompt_generations.append(result)
#         generations.append(prompt_generations)
    
#     model_ds.add_column('responses', generations)
#     all_ds.append(model_ds)

# generated_ds = concatenate_datasets(all_ds)
# pl.from_pandas(generated_ds.to_pandas()).write_parquet('data/generated.parquet')