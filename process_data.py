import polars as pl
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset, concatenate_datasets

train = (
    pl.read_csv('data/train.csv')
    .with_columns(
        prompt=pl.col('prompt').str.json_decode(),
        response_a=pl.col('response_a').str.json_decode(),
        response_b=pl.col('response_b').str.json_decode(),
        labels=pl.when(pl.col('winner_model_a') == 1)
            .then(0)
            .when(pl.col('winner_model_b') == 1)
            .then(1)
            .otherwise(2),
        fold=pl.lit(-1)
    )
)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

train = train.to_pandas()
for i, (train_idx, test_idx) in enumerate(skf.split(train, train['labels'])):
    train.loc[test_idx, 'fold'] = i

print(train['fold'].value_counts())

train = pl.from_pandas(train)
train.write_parquet('data/train.parquet')

test = (
    pl.read_csv('data/test.csv')
    .with_columns(
        prompt=pl.col('prompt').str.json_decode(),
        response_a=pl.col('response_a').str.json_decode(),
        response_b=pl.col('response_b').str.json_decode(),
    )
)

test.write_parquet('data/test.parquet')

ds = (
    pl.read_csv('data/lmsys-33k-deduplicated.csv')
    .with_row_index()
    .with_columns(
        prompt=pl.col('prompt').str.json_decode(),
        response_a=pl.col('response_a').str.json_decode(),
        response_b=pl.col('response_b').str.json_decode(),
        labels=pl.when(pl.col('winner_model_a') == 1)
            .then(0)
            .when(pl.col('winner_model_b') == 1)
            .then(1)
            .otherwise(2),
        fold=pl.lit(-1),
        id=pl.col('index')
    )
    .drop('index')
)

ds.write_parquet('data/lmsys-33k-deduplicated.parquet')

def transform_example(example):
    return {
        'prompt': [example['orig_instruction']],
        'response_a': [example['orig_response_A']],
        'response_b': [example['orig_response_B']],
        'labels': 0 if example['orig_score_A'] > example['orig_score_B'] else 1,
        'fold': -1
    }

def split_columns(row):
    conversation_a = row['chosen']
    conversation_b = row['rejected']

    prompt = []
    response_a = []
    for turn in conversation_a:
        if turn['role'] == 'user':
            prompt.append(turn['content'])
        else:
            response_a.append(turn['content'])
    
    response_b = []
    for turn in conversation_b:
        if turn['role'] == 'assistant':
            response_b.append(turn['content'])
    
    return {
        'prompt': prompt,
        'response_a': response_a,
        'response_b': response_b
    }

ds = load_dataset('mlabonne/orpo-dpo-mix-40k', split='train')
ds = ds.map(split_columns, num_proc=12)
ds = pl.from_pandas(ds.to_pandas())
ds = ds.with_columns(fold=pl.lit(-1))
ds.write_parquet('data/orpo-dpo-mix-40k.parquet')
