import polars as pl

from sklearn.model_selection import StratifiedKFold

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