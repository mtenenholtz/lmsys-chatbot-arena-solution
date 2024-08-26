accelerate launch llm_qlora.py -C configs/pair_pref.yaml
accelerate launch llm_validate.py -C configs/pair_pref.yaml
accelerate launch llm_validate.py -C configs/pair_pref.yaml --tta
accelerate launch llm_pseudo_label.py -C configs/pair_pref.yaml
accelerate launch llm_pseudo_label.py -C configs/pair_pref.yaml --tta

accelerate launch llm_qlora.py -C configs/gemma_rm.yaml
accelerate launch llm_validate.py -C configs/gemma_rm.yaml
accelerate launch llm_validate.py -C configs/gemma_rm.yaml --tta
accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml
accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml --tta

accelerate launch llm_qlora.py -C configs/gemma_rm_no_cap.yaml
accelerate launch llm_validate.py -C configs/gemma_rm_no_cap.yaml
accelerate launch llm_validate.py -C configs/gemma_rm_no_cap.yaml --tta
accelerate launch llm_pseudo_label.py -C configs/gemma_rm_no_cap.yaml
accelerate launch llm_pseudo_label.py -C configs/gemma_rm_no_cap.yaml --tta

python oof_optimization.py

accelerate launch llm_validate.py -C configs/gemma_rm_pseudo_rd_2.yaml
accelerate launch llm_validate.py -C configs/gemma_rm_pseudo_rd_2.yaml --tta