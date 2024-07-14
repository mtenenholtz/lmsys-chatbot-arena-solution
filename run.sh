# accelerate launch llm_qlora.py -C configs/internlm.yaml
# accelerate launch llm_qlora.py -C configs/mistral.yaml
# accelerate launch llm_qlora.py -C configs/qwen2.yaml
# accelerate launch llm_qlora.py -C configs/zephyr.yaml
# accelerate launch llm_qlora.py -C configs/glm.yaml
# accelerate launch llm_qlora.py -C configs/llama_3_base.yaml
# accelerate launch llm_qlora.py -C configs/mistral_base.yaml
# accelerate launch llm_qlora.py -C configs/qwen2_base.yaml

# accelerate launch llm_pseudo_label.py -C configs/llama_3_base.yaml
# accelerate launch llm_pseudo_label.py -C configs/mistral_base.yaml
# accelerate launch llm_pseudo_label.py -C configs/qwen2_base.yaml
# accelerate launch llm_pseudo_label.py -C configs/llama_3_base.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/mistral_base.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/qwen2_base.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/llama_3.yaml
# accelerate launch llm_pseudo_label.py -C configs/llama_3.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/internlm.yaml
# accelerate launch llm_pseudo_label.py -C configs/internlm.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/mistral.yaml
# accelerate launch llm_pseudo_label.py -C configs/mistral.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/qwen2.yaml
# accelerate launch llm_pseudo_label.py -C configs/qwen2.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/internlm.yaml
# accelerate launch llm_pseudo_label.py -C configs/internlm.yaml --tta
# accelerate launch llm_pseudo_label.py -C configs/glm.yaml
# accelerate launch llm_pseudo_label.py -C configs/glm.yaml --tta

# accelerate launch llm_validate.py -C configs/llama_3_base.yaml
# accelerate launch llm_validate.py -C configs/mistral_base.yaml
# accelerate launch llm_validate.py -C configs/qwen2_base.yaml
# accelerate launch llm_validate.py -C configs/llama_3_base.yaml --tta
# accelerate launch llm_validate.py -C configs/mistral_base.yaml --tta
# accelerate launch llm_validate.py -C configs/qwen2_base.yaml --tta

# accelerate launch llm_validate.py -C configs/internlm.yaml
# accelerate launch llm_validate.py -C configs/internlm.yaml --tta
# accelerate launch llm_validate.py -C configs/mistral.yaml
# accelerate launch llm_validate.py -C configs/mistral.yaml --tta
# accelerate launch llm_validate.py -C configs/qwen2.yaml
# accelerate launch llm_validate.py -C configs/qwen2.yaml --tta
# accelerate launch llm_validate.py -C configs/zephyr.yaml
# accelerate launch llm_validate.py -C configs/zephyr.yaml --tta
# accelerate launch llm_validate.py -C configs/glm.yaml
# accelerate launch llm_validate.py -C configs/glm.yaml --tta

# python oof_optimization.py

# accelerate launch llm_qlora.py -C configs/gemma.yaml
# accelerate launch llm_validate.py -C configs/gemma.yaml
# accelerate launch llm_validate.py -C configs/gemma.yaml --tta

accelerate launch llm_pseudo_label.py -C configs/pair_pref.yaml
accelerate launch llm_pseudo_label.py -C configs/pair_pref.yaml --tta
python oof_optimization.py
accelerate launch llm_train_pseudo.py -C configs/pair_pref_pseudo.yaml
accelerate launch llm_validate.py -C configs/pair_pref_pseudo.yaml
accelerate launch llm_validate.py -C configs/pair_pref_pseudo.yaml --tta