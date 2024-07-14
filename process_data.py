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

ds = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', split='train')

ds = (
    pl.from_pandas(ds.to_pandas())
    .with_row_index()
    # .drop('chosen', 'rejected', 'score_chosen', 'score_rejected')
    .with_columns(
        response_a=pl.when(pl.col('index') % 2 == 0).then(pl.col('chosen')).otherwise(pl.col('rejected')),
        response_b=pl.when(pl.col('index') % 2 == 0).then(pl.col('rejected')).otherwise(pl.col('chosen')),
    )
    .with_columns(
        response_a=pl.col('response_a').list.get(0).struct.field('content'),
        response_b=pl.col('response_b').list.get(0).struct.field('content'),
    )
    .select([
        pl.col('index').cast(pl.Int32).alias('id'),
        pl.col('prompt'),
        pl.col('response_a'),
        pl.col('response_b'),
    ])
)
ds = (
    ds
    .with_columns(
        prompt=pl.Series([[p] for p in ds['prompt'].to_list()]),
        response_a=pl.Series([[p] for p in ds['response_a'].to_list()]),
        response_b=pl.Series([[p] for p in ds['response_b'].to_list()]),
    )
    .select([
        pl.col('prompt'),
        pl.col('response_a'),
        pl.col('response_b'),
        pl.lit(-1).alias('fold')
    ])
)

ds.write_parquet('data/ultrafeedback.parquet')

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

# ds = load_dataset('prometheus-eval/Preference-Collection')['train']
# ds = (
#     ds
#     .map(transform_example, num_proc=12)
# )
# df = pd.DataFrame.from_records(ds.flatten())
# df = df.loc[(df.orig_score_A == '5') | (df.orig_score_B == '5')].reset_index(drop=True)
# df['fold'] = -1
# df = df[['prompt', 'response_a', 'response_b', 'labels', 'fold']]
# df.to_parquet('data/preference_collection.parquet')

ds = load_dataset('lmsys/mt_bench_human_judgments', split='human')

def split_columns(row):
    conversation_a = row['conversation_a'][:2]
    conversation_b = row['conversation_b'][:2]

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

ds = ds.filter(lambda x: x['turn'] == 1)
ds = ds.map(split_columns, num_proc=12)
ds = pl.from_pandas(ds.to_pandas())
ds = (
    ds
    .with_row_index()
    .with_columns(
        labels=pl.when(pl.col('winner') == 'model_a')
            .then(0)
            .when(pl.col('winner') == 'model_b')
            .then(1)
            .otherwise(2),
        fold=pl.lit(-1),
        id=pl.col('index')
    )
    .drop('index')
)

ds.write_parquet('data/mt_bench_human_judgments.parquet')

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

ds = load_dataset('nvidia/HelpSteer2', split='train')
df = ds.to_pandas()
df = df[~df.prompt.str.contains('<extra_id_1>')].reset_index(drop=True)
df['avg_score'] = df[['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']].mean(axis=1)
df = df.sort_values('avg_score', ascending=False).groupby('prompt').head(2)
df_a = df[['prompt', 'response']].groupby('prompt').head(1)
df_b = df[['prompt', 'response']].groupby('prompt').tail(1)
df = df_a.merge(df_b, on='prompt', suffixes=('_a', '_b'))
df = df.assign(labels=0, fold=-1)
df = df[~(df.response_a == df.response_b)].reset_index(drop=True)
df = df.assign(
    prompt=df.prompt.apply(lambda p: [p]),
    response_a=df.response_a.apply(lambda p: [p]),
    response_b=df.response_b.apply(lambda p: [p]),
)
pl.from_pandas(df).write_parquet('data/helpsteer2.parquet')
print(pl.from_pandas(df))
print((df.response_a == df.response_b).sum())

import random
random.seed(34)

def prompts(example):
    if random.random() < 0.5:
        return {
            'prompt': [example['prompt']],
            'response_a': [example['chosen']],
            'response_b': [example['rejected']],
        }
    else:
        return {
            'prompt': [example['prompt']],
            'response_a': [example['rejected']],
            'response_b': [example['chosen']],
        }
        
ds = load_dataset('jondurbin/py-dpo-v0.1', split='train')
ds = ds.map(prompts, num_proc=12, remove_columns=['chosen', 'rejected'])
ds = pl.from_pandas(ds.to_pandas())
ds = ds.with_columns(fold=pl.lit(-1), labels=pl.lit(0))
print(ds)
ds.write_parquet('data/pydpo.parquet')

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

ds = load_dataset('argilla/Capybara-Preferences', split='train')
ds = ds.map(split_columns, num_proc=12)
ds = pl.from_pandas(ds.to_pandas())
ds = ds.with_columns(fold=pl.lit(-1))
ds.write_parquet('data/Capybara-Preferences.parquet')

def transform_example(example):
    if random.random() < 0.5:
        return {
            'prompt': [example['input']],
            'response_a': [example['chosen']],
            'response_b': [example['rejected']],
            'labels': 0,
            'fold': -1
        }
    else:
        return {
            'prompt': [example['input']],
            'response_a': [example['rejected']],
            'response_b': [example['chosen']],
            'labels': 0,
            'fold': -1
        }

random.seed(34)
ds = load_dataset('argilla/distilabel-intel-orca-dpo-pairs', split='train')
ds = ds.map(transform_example, num_proc=12)
ds = pl.from_pandas(ds.to_pandas())
ds = ds.with_columns(fold=pl.lit(-1))
ds.write_parquet('data/distilabel-intel-orca-dpo-pairs.parquet')