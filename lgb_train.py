import lightgbm as lgb
import pandas as pd
import numpy as np

import re

from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

def stringify(c):
    values = [c for c in c if c is not None]
    return ' '.join(values)

def stringify_multiple(row):
    return ' '.join([stringify(row[c]) for c in ['prompt', 'response_a', 'response_b']])

df = pd.read_parquet('data/train.parquet')
# pred_map = {
#     'llama': Dataset.from_parquet('data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_lmsys_lr_1e-4_no_drop_qkvo_r64.parquet').to_pandas(),
#     'deberta': Dataset.from_parquet('data/preds/OpenAssistant_reward-model-deberta-v3-large-v2-sliding_window_1024_overlap_lstm_no_drop.parquet').to_pandas()
# }
# for i, (k, v) in enumerate(pred_map.items()):
#     pred_df = v[['id', 'prob_model_a', 'prob_model_b', 'prob_tie']]
#     pred_df = pred_df.rename(columns={f'prob_{c}': f'prob_{c}_{k}' for c in ['model_a', 'model_b', 'tie']})
#     if i == 0:
#         preds = pred_df
#     else:
#         preds = preds.merge(pred_df, on='id', how='left')

df = (
    df
    # .merge(preds, on='id', how='left')
    .assign(
        prompt_wc=lambda x: x.prompt.apply(stringify).str.split(' ').str.len(),
        response_a_wc=lambda x: x.response_a.apply(stringify).str.split(' ').str.len(),
        response_b_wc=lambda x: x.response_a.apply(stringify).str.split(' ').str.len(),
        num_turns=lambda x: x.prompt.str.len(),
    )
)

features = []
features += [
    'prompt_wc', 'response_a_wc', 'response_b_wc', 'num_turns'
]

count_feature_tokens = ['\n', '\n\n', '.', ' ', '","']
for i, e in enumerate(count_feature_tokens):
    for c in ['prompt', 'response_a', 'response_b']:
        feature_name = f'count_{c}_{i}'
        df[feature_name] = df[c].str.count(e).values
        features.append(feature_name)

vector_fit_text = df[['prompt', 'response_a', 'response_b']].apply(stringify_multiple, axis=1)

print('Fitting vectorizer')
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 7),
    analyzer='char',
    lowercase=False,
    min_df=10,
    max_df=0.95,
    sublinear_tf=True
)

tfidf_vectorizer = tfidf_vectorizer.fit(vector_fit_text)

columns_to_vectorize = ['prompt', "response_a", "response_b"]

print('Vectorizing columns')
vectorized_columns = []
for column in columns_to_vectorize:
    vectorized_columns.append(tfidf_vectorizer.transform(df[column].apply(stringify)))
# combined_train_tfidf = hstack(vectorized_columns).astype(np.float32).todense()
combined_train_tfidf = hstack(vectorized_columns)
svd = TruncatedSVD(n_components=32, random_state=42)
combined_train_tfidf = svd.fit_transform(combined_train_tfidf)

tfidf_features = [f'tfidf_{i}' for i in range(combined_train_tfidf.shape[1])] 
tfidf_df = pd.DataFrame(combined_train_tfidf, columns=tfidf_features)

df = pd.concat([
    df,
    tfidf_df
], axis=1)

# features = [c for c in preds.columns if c != 'id']
features += tfidf_features

params = dict(
    objective='multiclass',
    num_classes=3,
    learning_rate=0.05,
    feature_fraction=0.5,
    subsample=0.5,
    min_samples_leaf=100
)

print('Fitting...')
for i in range(4):
    train = df[df['fold'] != i]
    val = df[df['fold'] == i]

    train_features = train[features].values
    train_dset = lgb.Dataset(
        train_features,
        train['labels']
    )

    val_features = val[features].values
    val_dset = lgb.Dataset(
        val_features,
        val['labels']
    )

    model = lgb.train(
        params,
        train_dset,
        num_boost_round=1000,
        valid_sets=[val_dset],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
    )

    # model.save_model(f'models/lgb_model_{i}.txt')