import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaPreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

def causal_pooling(input_ids, x, mask, pad_token_id):
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    sequence_lengths = sequence_lengths.to(x.device)

    x = x[torch.arange(x.shape[0], device=x.device), sequence_lengths]

    return x

class BiEncoder(LlamaPreTrainedModel):
    def __init__(self, config, model_name, quant_config):
        super().__init__(config)
        self.transformer = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            pad_token_id=config.pad_token_id,
            attn_implementation='flash_attention_2'
        )
        self.transformer.config.pad_token_id = config.pad_token_id
        
        self.head = nn.Linear(config.hidden_size * 2, 3)

    def forward(self, input_ids, attention_mask, input_ids_b, attention_mask_b, labels=None, **kwargs):
        x_a = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        x_a = causal_pooling(input_ids, x_a, attention_mask, self.transformer.config.pad_token_id)
        
        x_b = self.transformer(
            input_ids=input_ids_b, 
            attention_mask=attention_mask_b
        ).last_hidden_state
        x_b = causal_pooling(input_ids_b, x_b, attention_mask_b, self.transformer.config.pad_token_id)

        x = torch.cat([x_a, x_b], dim=-1)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

