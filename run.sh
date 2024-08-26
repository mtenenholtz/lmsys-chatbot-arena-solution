

# accelerate launch llm_qlora.py -C configs/qwen2.yaml
# accelerate launch llm_validate.py -C configs/qwen2.yaml
# accelerate launch llm_validate.py -C configs/qwen2.yaml --tta

accelerate launch llm_validate.py -C configs/gemma_rm_pseudo_rd_2.yaml
accelerate launch llm_validate.py -C configs/gemma_rm_pseudo_rd_2.yaml --tta

# accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml
# accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml --tta
# python oof_optimization.py

# accelerate launch llm_pseudo_label.py -C configs/mistral_nemo.yaml
# accelerate launch llm_pseudo_label.py -C configs/mistral_nemo.yaml --tta

# accelerate launch llm_pseudo_label.py -C configs/pair_pref.yaml
# accelerate launch llm_pseudo_label.py -C configs/pair_pref.yaml --tta
# python oof_optimization.py
# accelerate launch llm_train_pseudo.py -C configs/gemma_rm_pseudo.yaml
# accelerate launch llm_validate.py -C configs/pair_pref_pseudo.yaml
# accelerate launch llm_validate.py -C configs/pair_pref_pseudo.yaml --tta