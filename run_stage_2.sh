accelerate launch llm_train_pseudo.py -C configs/gemma_rm_pseudo.yaml
accelerate launch llm_validate.py -C configs/pair_pref_pseudo.yaml
accelerate launch llm_validate.py -C configs/pair_pref_pseudo.yaml --tta