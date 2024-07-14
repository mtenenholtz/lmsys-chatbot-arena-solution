from transformers import AutoConfig, AutoModel, LlamaPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput
from peft import prepare_model_for_kbit_training

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

class SlidingWindowTransformerModel(nn.Module):
    def __init__(self, model_name, window_size=1800):
        super(SlidingWindowTransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(model_name)
        config_model.update({
            'hidden_dropout_prob': 0.,
            'attention_probs_dropout_prob': 0.,
            'max_position_embeddings': 1024
        })

        self.transformer = AutoModel.from_pretrained(model_name, config=config_model)

        self.lstm = ResidualLSTM(config_model.hidden_size)
        self.pooler = MeanPooling()
        self.head = nn.Linear(config_model.hidden_size, 3)
        
        self.window_size = 1024
        self.edge_len = 128
        self.inner_len = self.window_size - self.edge_len*2

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        x = input_ids
        B, L = input_ids.shape
        if L <= self.window_size:
            x = self.transformer(x, attention_mask=attention_mask, return_dict=False)[0]
        else:
            segments=(L-self.window_size)//self.inner_len
            if (L-self.window_size)%self.inner_len>self.edge_len:
                segments+=1
            elif segments==0:
                segments+=1
            x_new=self.transformer(x[:,:self.window_size],attention_mask=attention_mask[:,:self.window_size],return_dict=False)[0]

            for i in range(1,segments+1):
                start=self.window_size-self.edge_len+(i-1)*self.inner_len
                end=self.window_size-self.edge_len+(i-1)*self.inner_len+self.window_size
                end=min(end,L)
                x_next=x[:,start:end]
                mask_next=attention_mask[:,start:end]
                x_next=self.transformer(x_next,attention_mask=mask_next,return_dict=False)[0]
                #L_next=x_next.shape[1]-self.edge_len,
                if i==segments:
                    x_next=x_next[:,self.edge_len:]
                else:
                    x_next=x_next[:,self.edge_len:self.edge_len+self.inner_len]
                #print(x_next.shape)
                x_new=torch.cat([x_new,x_next],1)
            x=x_new
        
        x = self.lstm(x)
        output = self.pooler(x, attention_mask)
        
        logits = self.head(output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    
def causal_pooling(input_ids, x, mask, pad_token_id):
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    sequence_lengths = sequence_lengths.to(x.device)

    x = x[torch.arange(x.shape[0], device=x.device), sequence_lengths]
    x = x.unsqueeze(1)

    return x

class SlidingWindowLLM(LlamaPreTrainedModel):
    def __init__(self, config, model_name, quant_config, pad_token_id):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            pad_token_id=pad_token_id,
            attn_implementation='flash_attention_2',
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.pad_token_id = pad_token_id

        self.lstm = ResidualLSTM(self.config.hidden_size)
        self.pooler = MeanPooling()
        self.score = nn.Linear(self.config.hidden_size, 3)
        self.window_size = 1024
        self.edge_len = 128
        self.inner_len = self.window_size - self.edge_len*2

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # x = input_ids
        # B, L = input_ids.shape
        # if L <= self.window_size:
        #     x = self.transformer(x, attention_mask=attention_mask, return_dict=False)[0]
        #     x = causal_pooling(input_ids, x, attention_mask, self.config.pad_token_id)
        # else:
        #     segments=(L-self.window_size)//self.inner_len
        #     if (L-self.window_size)%self.inner_len>self.edge_len:
        #         segments+=1
        #     elif segments==0:
        #         segments+=1
        #     x_new=self.transformer(x[:,:self.window_size],attention_mask=attention_mask[:,:self.window_size],return_dict=False)[0]
        #     x_new = causal_pooling(x[:,:self.window_size], x_new, attention_mask[:,:self.window_size], self.config.pad_token_id)

        #     for i in range(1,segments+1):
        #         start=self.window_size-self.edge_len+(i-1)*self.inner_len
        #         end=self.window_size-self.edge_len+(i-1)*self.inner_len+self.window_size
        #         end=min(end,L)
        #         x_next=x[:,start:end]
        #         mask_next=attention_mask[:,start:end]
        #         x_next=self.transformer(x[:, start:end],attention_mask=mask_next,return_dict=False)[0]
        #         x_next = causal_pooling(x[:, start:end], x_next, mask_next, self.config.pad_token_id) 
        #         #L_next=x_next.shape[1]-self.edge_len,
        #         x_new=torch.cat([x_new,x_next],1)
        #     x=x_new
        x = input_ids
        B, L = input_ids.shape
        if L <= self.window_size:
            x = self.model(x, attention_mask=attention_mask, return_dict=False)[0]
        else:
            segments=(L-self.window_size)//self.inner_len
            if (L-self.window_size)%self.inner_len>self.edge_len:
                segments+=1
            elif segments==0:
                segments+=1
            x_new=self.model(x[:,:self.window_size],attention_mask=attention_mask[:,:self.window_size],return_dict=False)[0]

            for i in range(1,segments+1):
                start=self.window_size-self.edge_len+(i-1)*self.inner_len
                end=self.window_size-self.edge_len+(i-1)*self.inner_len+self.window_size
                end=min(end,L)
                x_next=x[:,start:end]
                mask_next=attention_mask[:,start:end]
                x_next=self.model(x_next,attention_mask=mask_next,return_dict=False)[0]
                #L_next=x_next.shape[1]-self.edge_len,
                if i==segments:
                    x_next=x_next[:,self.edge_len:]
                else:
                    x_next=x_next[:,self.edge_len:self.edge_len+self.inner_len]
                #print(x_next.shape)
                x_new=torch.cat([x_new,x_next],1)
            x=x_new

        x = self.lstm(x)

        output = x.mean(1)
        logits = self.score(output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )