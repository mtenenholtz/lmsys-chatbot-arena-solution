from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'microsoft/Phi-3-medium-128k-instruct'
quant_path = '/mnt/one/awq_models/microsoft/Phi-3-medium-128k-instruct'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)