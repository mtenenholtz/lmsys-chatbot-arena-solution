from transformers import LlamaPreTrainedModel, LlamaModel, AutoModel, Phi3Model
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import prepare_model_for_kbit_training, PeftModel

from models.modeling_phi3_small import Phi3SmallModel
from models.modeling_internlm2 import InternLM2Model

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//4)
        self.LSTM=nn.GRU(d_model//4, d_model//4, num_layers=2, bidirectional=True, dropout=0.)
        self.dropout1=nn.Dropout(0.)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x)

class Model(LlamaPreTrainedModel):
    def __init__(self, config, model_name, quant_config, pad_token_id, training=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        if 'internlm' in model_name:
            self.model = InternLM2Model.from_pretrained(
                model_name,
                quantization_config=quant_config,
                pad_token_id=pad_token_id,
                attn_implementation='flash_attention_2',
                trust_remote_code=True,
            )
        elif 'armorm' in model_name:
            self.model = LlamaModel.from_pretrained(
                model_name,
                quantization_config=quant_config,
                pad_token_id=pad_token_id,
                attn_implementation='flash_attention_2',
                trust_remote_code=True,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                num_labels=3,
                quantization_config=quant_config,
                pad_token_id=pad_token_id,
                attn_implementation='flash_attention_2',
                trust_remote_code=True,
            )
        # self.model = PeftModel.from_pretrained(self.model, '/mnt/one/kaggle/lmsys-chatbot-arena/meta-llama/Meta-Llama-3-8B-Instruct-sft_len_1800_surround-fold-0/checkpoint-4019/')
        # self.model.merge_and_unload()
        # self.model = self.model.base_model

        if training:
            self.model = prepare_model_for_kbit_training(self.model)
        self.lstm = ResidualLSTM(config.hidden_size)
        self.score = nn.Linear(config.hidden_size, 3, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.lstm(hidden_states)
        logits = self.score(hidden_states)

        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)

        logits = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return SequenceClassifierOutput(
            logits=logits,
            loss=loss,
        )

    def token_dropout(self, input_ids, p_token_mask):
        if p_token_mask <= 0.:
            return input_ids

        special_mask = torch.ones_like(input_ids)
        special_mask[
            (input_ids == self.model.config.bos_token_id)
            | (input_ids == self.model.config.eos_token_id)
        ] = 0
        mask = (
            torch.bernoulli(
                torch.full(input_ids.shape, p_token_mask)
            )
            .to(input_ids.device)
            .bool()
            & special_mask
        ).bool()
        input_ids[mask] = self.model.config.eos_token_id

        return input_ids

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings