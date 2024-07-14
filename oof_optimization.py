# %%
from scipy.optimize import minimize
from time import time
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pprint import pprint

import pandas as pd
import numpy as np
import polars as pl
import datetime
import os

# %%
def loss(y_pred):
    return log_loss(y_true, y_pred)

def metric(weights):
    oof_blend = np.tensordot(weights, oof, axes = ((0), (0)))
    score = loss(oof_blend)
    return score

# %%
oof_dict = {
    # 'model_1': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_lstm_2_layer.parquet',
    # 'model_2': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_lstm_2_layer_tta.parquet',
    # 'model_3': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround.parquet',
    # 'model_4': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround_tta.parquet',
    # 'model_1': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround_gn_100.parquet',
    # 'model_2': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround_gn_100_tta.parquet',
    # 'model_5': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround_start_trunc.parquet',
    # 'model_6': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround_start_trunc_tta.parquet',
    # 'model_1': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround_diff_lr_1e-5.parquet',
    # 'model_2': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_surround_diff_lr_1e-5_tta.parquet',
    'model_1': 'data/preds/RLHFlow_pair-preference-model-LLaMA3-8B-llm_surround.parquet',
    'model_2': 'data/preds/RLHFlow_pair-preference-model-LLaMA3-8B-llm_surround_tta.parquet',
    # 'model_1': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_pseudo_orpo_soft_labels_opt.parquet',
    # 'model_2': 'data/preds/meta-llama_Meta-Llama-3-8B-Instruct-llm_pseudo_orpo_soft_labels_opt_tta.parquet',
    # 'model_3': 'data/preds/mistralai_Mistral-7B-Instruct-v0.3-llm_surround.parquet',
    # 'model_4': 'data/preds/mistralai_Mistral-7B-Instruct-v0.3-llm_surround_tta.parquet',
    # 'model_5': 'data/preds/Qwen_Qwen2-7B-Instruct-llm_surround.parquet',
    # 'model_6': 'data/preds/Qwen_Qwen2-7B-Instruct-llm_surround_tta.parquet',
    # 'model_7': 'data/preds/HuggingFaceH4_zephyr-7b-beta-llm_surround.parquet',
    # 'model_8': 'data/preds/HuggingFaceH4_zephyr-7b-beta-llm_surround_tta.parquet',
    # 'model_9': 'data/preds/internlm_internlm2_5-7b-chat-llm_surround.parquet',
    # 'model_10': 'data/preds/internlm_internlm2_5-7b-chat-llm_surround_tta.parquet',
}



# %%
targets = ['prob_model_a', 'prob_model_b', 'prob_tie']
df = pd.read_parquet('data/train.parquet')
dfs = [df.merge(pd.read_parquet(f)[['id']], on='id') for f in oof_dict.values()]
df = min([d for d in dfs], key=lambda x: x.shape[0])
y_true = df['labels'].values

# %%
oof_dfs = [df.merge(pd.read_parquet(f), on='id')[targets] for f in oof_dict.values()]
oof = np.zeros((len(oof_dict), y_true.shape[0], 3))
for i, _df in enumerate(oof_dfs):
    oof[i] = _df[targets].values

# %%
metric_scores = {}
for n, key in enumerate(oof_dict.keys()):
    score_oof = loss(oof[n])
    metric_scores[key] = score_oof
    print(f'{key} CV: {score_oof:.6f}')

# %%
tol = 1e-10
init_guess = [1 / oof.shape[0]] * oof.shape[0]
bnds = [(0, 1) for _ in range(oof.shape[0])]
cons = {'type': 'eq', 
        'fun': lambda x: np.sum(x) - 1, 
        'jac': lambda x: [1] * len(x)}

print(f'Inital Blend OOF: {metric(init_guess):.6f}', )

# %%
start_time = time()

res_scipy = minimize(fun = metric, 
                     x0 = init_guess, 
                     method = 'Powell', 
                     #method='SLSQP',
                     bounds = bnds, 
                     options=dict(maxiter=1_000_000),
                     tol = tol)

print(f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}] Optimised Blend OOF: {res_scipy.fun:.6f}')
print(f'Optimised Weights: {res_scipy.x}')
print('-' * 70)

for n, key in enumerate(oof_dict.keys()):
    print(f'{key} Optimised Weights: {res_scipy.x[n]:.6f}')

ws = [ res_scipy.x[i] for i in range(len(oof_dict.keys()))]
print(f'Normalized weights:')
weights = ws / np.sum(ws)

# %%
weight_dict = {}
for i, (k, v) in enumerate(oof_dict.items()):
    model_name = v.split('/')[-1].split('.csv')[0]
    weight_dict[model_name] = weights[i]

pprint(weight_dict)

# %%
opt_oofs = np.array([df.merge(pd.read_parquet(f'data/preds/{f}'), on='id')[targets].values * w for f, w in weight_dict.items()])
opt_oofs = opt_oofs.sum(0)
pd.DataFrame({
    c: opt_oofs[:, i] for i, c in enumerate(targets)
}).to_parquet('data/preds/opt_oofs.parquet')

# %%
def fname(f, fold):
    return f.replace('.parquet', f'_fold_{fold}.parquet')

pseudo_targets = ['winner_model_a_pred', 'winner_model_b_pred', 'winner_tie_pred']

base_file = list(weight_dict.keys())[0]
base_df = pd.read_parquet(f'data/pseudo/{fname(base_file, 0)}').sort_values('id')[['id']]
for i in range(1):
    opt_pseudos = np.array([base_df.merge(pd.read_parquet(f'data/pseudo/{fname(f, i)}'), on='id')[pseudo_targets].values * w for f, w in weight_dict.items()])
    opt_pseudos = opt_pseudos.sum(0)
    pseudos = pd.DataFrame({
        'id': pd.read_parquet(f'data/pseudo/{fname(list(weight_dict.keys())[0], i)}').sort_values('id').id,
    })
    
    for idx, c in enumerate(pseudo_targets):
        pseudos[c] = opt_pseudos[:, idx]
    pseudos.reset_index(drop=True).to_parquet(f'data/pseudo/opt_pseudos_fold_{i}.parquet')

# %%
winner_a = (pl.col('winner_model_a_pred') > pl.col('winner_model_b_pred')) & (pl.col('winner_model_a_pred') > pl.col('winner_tie_pred'))
winner_b = (pl.col('winner_model_b_pred') > pl.col('winner_model_a_pred')) & (pl.col('winner_model_b_pred') > pl.col('winner_tie_pred'))
winner_tie = (pl.col('winner_tie_pred') > pl.col('winner_model_a_pred')) & (pl.col('winner_tie_pred') > pl.col('winner_model_b_pred'))

pseudos = (
    pl.read_parquet('data/pseudo/opt_pseudos_fold_0.parquet')
    .filter(pl.col('id') < 0)
    .with_columns(
        winner_model_a=pl.when(winner_a).then(1).otherwise(0),
        winner_model_b=pl.when(winner_b).then(1).otherwise(0),
        winner_tie=pl.when(winner_tie).then(1).otherwise(0),
    )
)

winner_a_df = pseudos.filter(pl.col('winner_model_a') == 1)#.sample(fraction=0.349/10, seed=42)
winner_b_df = pseudos.filter(pl.col('winner_model_b') == 1)#.sample(fraction=0.342/10, seed=42)
winner_tie_df = pseudos.filter(pl.col('winner_tie') == 1)#.sample(fraction=0.309/10, seed=42)

winner_df = pl.concat([
    winner_a_df,
    winner_b_df,
    winner_tie_df

])
winner_df.write_parquet('data/pseudo/opt_pseudos_fold_0_sampled.parquet')

# %%



