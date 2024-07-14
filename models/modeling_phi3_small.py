import math
from typing import Any, Dict, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.cache_utils import Cache, DynamicCache

from .triton_flash_blocksparse_attn import BlockSparseParams
from .triton_blocksparse_attention_layer import BlockSparseAttentionLayer
from .positional_embedding import RotaryEmbedding

from .configuration_phi3_small import Phi3SmallConfig

# Flash Attention Related Imports
is_flash_attention_available = False
try:
    import flash_attn
    if int(flash_attn.__version__.split('.')[0]) < 2:
        from flash_attn.flash_attn_interface import (
            flash_attn_func,
            flash_attn_unpadded_kvpacked_func as flash_attn_varlen_kvpacked_func,
            )

        # rename `max_seqlen`
        def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, **kwargs):
            return flash_attn_func(qkv, cu_seqlens, dropout_p=dropout_p, max_s=max_seqlen, **kwargs)

    else:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_kvpacked_func,
        )
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    is_flash_attention_available = True
except ImportError:
    pass

logger = logging.get_logger(__name__)

LegacyCache = Tuple[Tuple[torch.FloatTensor]]

# Taken from https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py
def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)
 
 
# Taken from https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py
def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@torch.jit.script
def gegelu(input, limit: Optional[float] = None):
    a_gelu, a_linear = input[..., ::2], input[..., 1::2]
    if limit is not None:
        a_gelu = torch.where(
            torch.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)
        )
        a_linear = torch.where(
            torch.isinf(a_linear), a_linear, a_linear.clamp(min=-limit, max=limit)
        )
    out_gelu = quick_gelu(a_gelu)
    return out_gelu * (a_linear + 1)

def collapse_first_n_dims(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Collapse the first `n` dimensions of a tensor into a single dimension.

    Args:
        x (torch.Tensor): The input tensor.
        n (int): The number of dimensions to collapse.

    Returns:
        torch.Tensor: The output tensor.
    """
    return x.view(-1, *x.shape[n:])

def pad_tensor_to_next_mult_of(
    tensor: torch.Tensor,
    dim: int,
    n: int,
) -> Tuple[torch.Tensor, int]:
    """
    Pads a tensor along a specified dimension to the next multiple of a given number.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension along which to pad the tensor.
        n (int): The number to pad the tensor to the next multiple of.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the padded tensor and the amount of padding added.
    """
    residual = tensor.size(dim) % n
    if residual == 0:
        return tensor, 0
    padding = n - residual
    padding_tensor = torch.zeros((*tensor.size()[:dim], padding, *tensor.size()[dim + 1:]), device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding_tensor], dim=dim), padding

def strip_padding_from_tensor(
    tensor: torch.Tensor,
    dim: int,
    residual: int,
) -> torch.Tensor:
    """
    Removes padding from a tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension along which to remove padding.
        residual (int): The amount of padding to remove.

    Returns:
        torch.Tensor: The tensor with padding removed along the specified dimension.
    """
    return torch.narrow(tensor, dim, 0, tensor.size(dim) - residual)

class Phi3SmallMLP(nn.Module):
    def __init__(self, config: Phi3SmallConfig):
        super().__init__()
        self.config = config
        assert self.config.hidden_act == "gegelu", "Only `gegelu` is supported for the Phi-3-small model .."
        self.hidden_size = config.hidden_size
        self.gegelu_limit = config.gegelu_limit
        self.intermediate_size = config.intermediate_size

        self.up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config.ffn_dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(
                gegelu(self.up_proj(x), limit=self.gegelu_limit)
            )
        )


class Phi3SmallSelfAttention(nn.Module):
    def __init__(self, config: Phi3SmallConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.hidden_size = config.hidden_size
        # Number of Query Heads
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # Number of Key Value Heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_q_per_kv = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_embedding_base = config.rope_embedding_base
        self.rope_position_scale = config.rope_position_scale
        self.is_causal = True

        self.attention_dropout_rate = config.attention_dropout_prob

        norm_factor = None
        if config.mup_use_scaling:
            norm_factor = self.head_dim / config.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(self.head_dim)
        self.softmax_scale = 1.0 / norm_factor

        self.query_key_value = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.blocksparse_params = None
        # layer_idx is 0 indexed because that's what the KV Cache expects.
        if self.config.dense_attention_every_n_layers and ((self.layer_idx + 1) % self.config.dense_attention_every_n_layers == 0):
            logger.info(
                f"Layer {layer_idx + 1} is using dense attention since it is divisible by "
                f"{self.config.dense_attention_every_n_layers}"
            )
            assert is_flash_attention_available, "Flash Attention is not available, but is needed for dense attention"
        else:
            # BlockSparse related Parameters
            self.blocksparse_params = BlockSparseParams.from_config(config)

        if self.blocksparse:
            active_head_range = None
            """
                ... note(bapatra)::

                    In case of tensor parallelism and while using the heterogeneous head patterns,
                    the active head range needs to be modified based on the tensor parallel rank
                    and the tensor parallel world size.

                    This is because in the case of heterogeneous head patterns, the kernel needs to know
                    which head is on which device, so that it can pick the corresponding blocksparse head
                    pattern correctly.

                    Example:
                    ```python

                        if not self.blocksparse_params.homo_head_pattern:
                            tp_rank = torch.distributed.get_rank() % tp_world_size
                            num_heads_per_partition = num_heads // tp_world_size
                            active_head_range = (tp_rank * num_heads_per_partition, (tp_rank + 1) * num_heads_per_partition)

                    ```

            """
            
            self._blocksparse_layer = BlockSparseAttentionLayer(
                n_heads=self.num_heads,
                max_seq_len=self.max_position_embeddings,
                sparse_block_size=self.blocksparse_params.block_size,
                local_blocks=self.blocksparse_params.num_local_blocks,
                vert_stride=self.blocksparse_params.vert_stride,
                kernel_block_size=self.blocksparse_params.kernel_block_size,
                homo_head=self.blocksparse_params.homo_head_pattern,
                active_head_range=active_head_range,
            )
        self.rotary_emb = RotaryEmbedding.from_config(config)


    @property
    def blocksparse(self):
        return self.blocksparse_params is not None

    def _split_heads(self, mixed_x_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, sq, _ = mixed_x_layer.size()
        r"""
        The main idea is that we group tensors as
        [bs, sq, (q00, q01, ... q0m, k0, v0), (q10, q11, ... q1m, k1, v1), ... (qn0, qn1, ... qnm, kn, vn)]
        That ways, when the MP column sharding happens, this tensor will be sharded keeping all the
        queries and keys intact. In order to get the correct qkv, we first break into groups, and then
        index into the groups.
        """

        intermediate_shape = (bs, sq, -1, (self.num_q_per_kv + 2), self.head_dim)
        mixed_x_layer = mixed_x_layer.view(*intermediate_shape)
        q = mixed_x_layer[:, :, :, :-2]
        k = mixed_x_layer[:, :, :, [-2]]
        v = mixed_x_layer[:, :, :, [-1]]
        q, k, v = [
            rearrange(
                x,
                "bs sq group nh hn -> bs sq (group nh) hn"
            ) for x in (q, k, v)
        ]
        return q, k, v

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._unpad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape


        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _apply_blocksparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.LongTensor],
        return_attention_probs: bool = False,
    ) -> torch.Tensor:
        """
        Applies blocksparse attention to the input tensors.

        Args:
            q (torch.Tensor): The query tensor of shape (bs, nqp, seq_len, hn).
            k (torch.Tensor): The key tensor of shape (bs, nkp, seq_len, hn).
            v (torch.Tensor): The value tensor of shape (bs, nkp, seq_len, hn).
            attention_mask (Optional[torch.LongTensor]): The attention mask tensor of shape (bs, seq_len).
            return_attention_probs (bool, optional): Whether to return attention probabilities. Defaults to False.

        Returns:
            torch.Tensor: The context layer tensor of shape (bs, nqp, seq_len, hn).
        """
        assert not return_attention_probs, "return_attention_probs is not supported for blocksparse attention"
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # shape: (bs, nqp, seq_len, hn)
        if torch.is_grad_enabled():
            # Training or non-batched inference
            context_layer = self._blocksparse_layer(
                q=q, k=k, v=v, sm_scale=self.softmax_scale
            )
        elif attention_mask is None:
            if q.size(0) != 1:
                logger.warning_once(
                    "You are attempting to do batched inference without passing the attention mask.\n"
                    "This is okay if you are running loglikelihood requests. However, if you want to do generation, "
                    "this probably won't work as expected. Please pass the attention mask to the forward function."
                )
            context_layer = self._blocksparse_layer(
                q=q, k=k, v=v, sm_scale=self.softmax_scale
            )
        else:
            """
                Shapes of tensors are as follows:
                    q: (bs, nqp, seq_len, hdim)
                    k: (bs, nkp, seq_len, hdim)
                    v: (bs, nkp, seq_len, hdim)
                We first need to transpose the shapes to fit what the
                kernel needs, and the reinvert it back at the end of the operations
            """
            assert attention_mask.ndim == 2, "The kernel, like flash-attention-2, only supports 2d attention masks ..."
            left_paddings = attention_mask.shape[1] - attention_mask.sum(dim=-1)
            # shape: (bs, seq_len, nqp, hdim)
            q = q.transpose(1, 2).contiguous()
            # shape: (bs, seq_len, nkp, hdim)
            k = k.transpose(1, 2).contiguous()
            # shape: (bs, seq_len, nkp, hdim)
            v = v.transpose(1, 2).contiguous()
            context_layer = self._blocksparse_layer(
                q=q, k=k, v=v, sm_scale=self.softmax_scale, left_paddings=left_paddings.to(torch.int32)
            )
            # shape: (bs, nqp, seq_len, hdim)
            context_layer = context_layer.transpose(1, 2).contiguous()
        return context_layer

    def _apply_dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply dense attention

        Args:
            q (torch.Tensor):
                The query tensor, shape: (bs, num_query_heads, seq_len, head_size)
            k (torch.Tensor):
                The key tensor, shape: (bs, num_query_heads, seq_len, head_size)
            v (torch.Tensor):
                The value tensor, shape: (bs, num_query_heads, seq_len, head_size)

            return_attention_probs (bool, optional):
                Return the attention probabilities. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Return the output of the attention aggregation. If `return_attention_probs` is True, then
                also return the attention probabilities

        .. note::
            Right now, am assuming the expansion for the query key values is already done
            outside. But ideally, since Flash attention handles the GQA correctly, we can
            avoid doing that.

        """
        attention_dropout_prob = self.attention_dropout_rate if self.training else 0.0
        # Get into the correct shape for the Flash Attention API
        # shape: (bs, seq_len, nqp, hn)
        q = q.transpose(1, 2).contiguous()
        query_length = q.size(1)
        # shape: (bs, seq_len, npq, hn)
        k = k.transpose(1, 2).contiguous()
        # shape: (bs, seq_len, npq, hn)
        v = v.transpose(1, 2).contiguous()

        if attention_mask is not None:
            causal = q.size(2) == k.size(2)
            batch_size = q.shape[0]
            flat_q, flat_k, flat_v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                q, k, v, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            flat_kv = torch.cat((flat_k.unsqueeze(1), flat_v.unsqueeze(1)), dim=1)
            attn_output_unpad = flash_attn_varlen_kvpacked_func(
                q=flat_q,
                kv=flat_kv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=attention_dropout_prob,
                softmax_scale=self.softmax_scale,
                causal=causal,
                return_attn_probs=return_attention_probs
            )
            attention_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            kv = torch.cat((k.unsqueeze(2), v.unsqueeze(2)), dim=2)
            cu_seqlens_q = torch.arange(
                0, (q.size(0) + 1), device=q.device, dtype=torch.int32
            ) * q.size(1)
            cu_seqlens_kv = torch.arange(
                0, (kv.size(0) + 1), device=kv.device, dtype=torch.int32
            ) * kv.size(1)
            max_seqlen_q = q.size(1)
            max_seqlen_k = kv.size(1)
            attention_output = flash_attn_varlen_kvpacked_func(
                q=collapse_first_n_dims(q, 2),
                kv=collapse_first_n_dims(kv, 2),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=attention_dropout_prob,
                softmax_scale=self.softmax_scale,
                causal=q.size(1) == kv.size(1),
                return_attn_probs=return_attention_probs
            )
        if return_attention_probs:
            (context_layer, attn_probs) = attention_output
            context_layer = context_layer.view(q.size(0), q.size(1), -1, q.size(3)).transpose(1, 2).contiguous()
            return (context_layer, attn_probs)
        context_layer = attention_output
        context_layer = context_layer.view(q.size(0), q.size(1), -1, q.size(3)).transpose(1, 2).contiguous()
        return context_layer

    
    def expand_kv_to_q_size(self, kv: torch.Tensor, num_q_per_kv: int) -> torch.Tensor:
        """
        Expand the key-value tensor to match the size of the query tensor.

        Args:
            kv (torch.Tensor): The key-value tensor of shape (bsz, nkp, 2, seq_len, hdim).
            num_q_per_kv (int): The number of queries per key-value.

        Returns:
            torch.Tensor: The expanded key-value tensor of shape (bsz, nqp, 2, seq_len, hdim).
            Where nqp = num_q_per_kv * nkp

        .. note(bapatra)::
            Right now, I am using a repeat_interleave to expand the kv to the size of q.
            This incurs a memory penalty, since the tensors are actually copied.
            TODO: If this does yield benefits, then potentially we can use the re-written
            flash attention kernel that can handle GQA.
        """

        repeats = torch.tensor([num_q_per_kv] * kv.size(1)).to(kv.device)
        total = repeats.sum()
        expanded_kv = torch.repeat_interleave(
            kv,
            repeats=repeats,
            dim=1,
            output_size=total
        )
        return expanded_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        The forward function of the Self Attention Layer.

        Args:
            hidden_states (torch.Tensor):
                The input tensor of shape (bs, q_len, h).
            attention_mask (Optional[torch.Tensor], optional):
                The attention mask tensor of shape (bs, seq_len). This is the 2D attention mask tensor as is standard in the flash-attention
                kernel.
                Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                The position ids tensor of shape (bs, q_len). Defaults to None. Unused by the function.
            past_key_value (Optional[Cache], optional): 
                The previous kv cache values. Defaults to None.
            output_attentions (bool, optional): 
                Whether to return the attention scores. Defaults to False.
                    .. note::
                        For the blocksparse attention kernel, we do not support returning the attention scores.
            use_cache (bool, optional): 
                Whether to use the cache for storing the kv. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                The output tensor of shape (bs, q_len, h), 
                the attention scores tensor of shape (bs, nqp, q_len, seq_len) if `output_attentions` is True, 
                and the updated cache values if `use_cache` is True.
        
        Notations:
        ------------
            bs: batch size
            sq_len: sequence length of the entire sequence
            q_len: sequence length of the query
            cache_sq: sequence length in the cache
                If there is no cache then cache_sq = 0
                and sq_len = q_len
                otherwise sq_len = q_len + cache_sq
            h: hidden size
            nq: number of query heads
            nkv: number of key heads
            hn: hidden size per head
                hn = h // nq
            nqp: number of query heads (per MP partition)
                nqp = nq // (num mp partitions)
            nkvp: number of key-value heads (per MP partition)
                nkvp = nk // (num mp partitions)

        """
        # shape: (bs, q_len, h)
        bsz, q_len, _ = hidden_states.size()

        # shape: (bs, q_len, (nqp + 2 * nkvp) * hn)
        mixed_x_layer = self.query_key_value(hidden_states)
        # shape: (bs, q_len, nqp, hn), shape: (bs, q_len, nkvp, hn), shape: (bs, q_len, nkvp, hn)
        q, k, v = self._split_heads(mixed_x_layer)

        # shape: (bs, qnp, q_len, hn)
        query_states = q.permute(0, 2, 1, 3).contiguous()
        # shape: (bs, nkvp, q_len, hn)
        key_states = k.permute(0, 2, 1, 3).contiguous()
        # shape: (bs, nkvp, q_len, hn)
        value_states = v.permute(0, 2, 1, 3).contiguous()

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            if self.rotary_emb is not None:
                seqlen_offset = past_key_values.get_usable_length(kv_seq_len, layer_idx=self.layer_idx)
                # shape: (bs, nqp, q_len, hn), shape: (bs, nkvp, q_len, hn)
                query_states, key_states = self.rotary_emb(
                    query_states, key_states, seq_dimension=2, seqlen_offset=seqlen_offset
                )
                key_states, value_states = past_key_values.update(key_states=key_states, value_states=value_states, layer_idx=self.layer_idx)
        else:
            # In this case seq_len = q_len and cache_sq = 0
            if self.rotary_emb is not None:
                # shape: (bs, nqp, seq_len, hn), shape: (bs, nkvp, seq_len, hn)
                query_states, key_states = self.rotary_emb(query_states, key_states, seq_dimension=2)

        # shape: (bs, nkvp, 2, seq_len, hn)
        kv_states = torch.cat((key_states.unsqueeze(2), value_states.unsqueeze(2)), dim=2)
        # shape: (bs, nqp, 2, seq_len, hn)
        expanded_kv_states = self.expand_kv_to_q_size(kv_states, num_q_per_kv=self.num_q_per_kv)
        # shape: (bs, nqp, seq_len, hn), shape: (bs, nqp, seq_len, hn)
        expanded_key_states, expanded_value_states = expanded_kv_states[:, :, 0], expanded_kv_states[:, :, 1]
        if self.blocksparse:
            attn_function_output = self._apply_blocksparse_attention(
                q=query_states,
                k=expanded_key_states,
                v=expanded_value_states,
                attention_mask=attention_mask,
                return_attention_probs=output_attentions
            )
        else:
            attn_function_output = self._apply_dense_attention(
                q=query_states,
                k=expanded_key_states,
                v=expanded_value_states,
                attention_mask=attention_mask,
                return_attention_probs=output_attentions
            )

        attn_weights = None
        if output_attentions:
            attn_output, attn_weights = attn_function_output
        else:
            # shape: (bs, nqp, seq_len, hn)
            attn_output = attn_function_output
        # shape: (bs, seq_len, nqp, hn)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # shape: (bs, seq_len, h)
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights, past_key_values
        

class Phi3SmallDecoderLayer(nn.Module):
    def __init__(self, config: Phi3SmallConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Phi3SmallSelfAttention(config, layer_idx)
        self.mlp = Phi3SmallMLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Cache]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_values,)

        return outputs



class Phi3SmallPreTrainedModel(PreTrainedModel):
    config_class = Phi3SmallConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3SmallDecoderLayer"]
    skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # The output projection on the decoder attention layer as well as the down_proj in the MLP are scaled
        # differently (dubbed `output_layer_init_method` in the Megatron code). This is replicated here
        for name, p in module.named_parameters():
            if any(x in name for x in ("c_proj.weight", "down_proj.weight", "o_proj.weight")):
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)))


class Phi3SmallModel(Phi3SmallPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Embedding Dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout_prob)
        
        # MuP Embedding scaling
        self.mup_embedding_multiplier = config.mup_embedding_multiplier

        self.layers = nn.ModuleList([Phi3SmallDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @property
    def pad_sequence_to_multiple_of_64(self):
        # We only need to do this for the backward pass. So only required
        # when we are in the context of generating gradients
        return self.config.pad_sequence_to_multiple_of_64 and torch.is_grad_enabled()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, LegacyCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.embedding_dropout(inputs_embeds)

        if self.mup_embedding_multiplier is not None and self.mup_embedding_multiplier > 0.0:
            inputs_embeds = inputs_embeds * self.mup_embedding_multiplier
        
        residual = 0
        if self.pad_sequence_to_multiple_of_64:
            # note(bapatra): Since we don't particularly use the position_ids and the attention mask
            # we don't need to pad them
            inputs_embeds, residual = pad_tensor_to_next_mult_of(tensor=inputs_embeds, dim=1, n=64)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                # Following the Mistral schema for layer return values
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        if residual > 0:
            hidden_states = strip_padding_from_tensor(tensor=hidden_states, dim=1, residual=residual)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi3SmallForCausalLM(Phi3SmallPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi3SmallModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.mup_width_multiplier = config.mup_width_multiplier

        # Create the mask for the dummy tokens in the vocabulary
        dummy_token_indices = config.dummy_token_indices
        dummy_tokens_mask = torch.zeros(self.vocab_size).bool()
        dummy_tokens_mask[dummy_token_indices] = True
        # shape: (vocab_size,)
        self.register_buffer("dummy_tokens_mask", dummy_tokens_mask, persistent=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,   
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        if self.mup_width_multiplier:
            logits = logits / self.mup_width_multiplier
        logits = logits.masked_fill(self.dummy_tokens_mask, min_value_of_dtype(logits.dtype))

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


# Copied from transformers.models.mistral.modeling_mistral.MistralForSequenceClassification with Mistral -> Phi3Small
class Phi3SmallForSequenceClassification(Phi3SmallPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Phi3SmallModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )