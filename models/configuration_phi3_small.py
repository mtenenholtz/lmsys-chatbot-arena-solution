# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from functools import cached_property

""" Phi3Small model configuration """
logger = logging.get_logger(__name__)


def next_mult(x, y):
    return (x + y - 1) // y * y

class Phi3SmallConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a `Phi3Small` model. It is used to
    instantiate a Phi-3-small model according to the specified arguments, defining the model architecture. 
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Phi-3-small
    [phi3](https://arxiv.org/pdf/2404.14219) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Phi3Small model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling `Phi3Small`.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might safely be used with.
        rope_embedding_base (`float`, *optional*, defaults to 10^6):
            The base value for the RoPE (Relative Position Encoding) embedding.
        rope_position_scale (`float`, *optional*, defaults to 1.0):
            The scale factor for the RoPE position encoding.
        rope_scaling (`Optional[Dict[str, Union[float, List[float], int]]]`, *optional*, defaults to None):
            The scaling configuration used for LongRoPE.
        hidden_size (`int`, *optional*, defaults to 4096):
            The size of the hidden layers in the model.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            The number of layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of query heads in the model.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            The number of key-value heads in the model.
        hidden_act (`str`, *optional*, defaults to "gegelu"):
            The activation function used in the model.
        gegelu_limit (`float`, *optional*, defaults to 20.0):
            The limit value for the GELU activation function (for numerical stability).
        gegelu_pad_to_256 (`bool`, *optional*, defaults to True):
            Whether to pad the intermediate size to a multiple of 256 (for faster matmul ops).
        ff_dim_multiplier (`Optional[int]`, *optional*, defaults to None):
            The dimension multiplier for the feed-forward layers.
        ff_intermediate_size (`Optional[int]`, *optional*, defaults to 14336):
            The intermediate size for the feed-forward layers.
            One of `ff_dim_multiplier` or `ff_intermediate_size` must be specified.
        blocksparse_homo_head_pattern (`bool`, *optional*, defaults to False):
            Whether to use a homogeneous head pattern for block-sparse attention.
        blocksparse_block_size (`int`, *optional*, defaults to 64):
            The block size for block-sparse attention.
        blocksparse_num_local_blocks (`int`, *optional*, defaults to 16):
            The number of local blocks for block-sparse attention.
            The local window used in blocksparse equals `blocksparse_num_local_blocks * blocksparse_block_size`
        blocksparse_vert_stride (`int`, *optional*, defaults to 8):
            The vertical stride for block-sparse attention.
        blocksparse_triton_kernel_block_size (`int`, *optional*, defaults to 64):
            The kernel block size for block-sparse attention.
        dense_attention_every_n_layers (`Optional[int]`, *optional*, defaults to 2):
            The frequency of all dense attention layers in the model
        embedding_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for the embedding layer.
        attention_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        ffn_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for the feed-forward layers.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon value for layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The range for weight initialization.
        mup_use_scaling (`bool`, *optional*, defaults to True):
            Whether to use scaling for MuP parameters (see: https://arxiv.org/abs/2203.03466).
        mup_width_multiplier (`bool`, *optional*, defaults to 8.0):
            The width multiplier for MuP.
        mup_embedding_multiplier (`bool`, *optional*, defaults to 10.0):
            The embedding multiplier for MuP.
        mup_attn_multiplier (`bool`, *optional*, defaults to 1.0):
            The attention multiplier for MuP.
        use_cache (`bool`, *optional*, defaults to True):
            Whether to use cache for the model.
        bos_token_id (`int`, *optional*, defaults to 100257):
            The token ID for the beginning of sentence.
        eos_token_id (`int`, *optional*, defaults to 100257):
            The token ID for the end of sentence.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to False):
            Whether to reorder and upcast attention.
        pad_sequence_to_multiple_of_64 (`bool`, *optional*, defaults to True):
            Whether to pad the sequence length to a multiple of 64.
        **kwargs:
            Additional keyword arguments.

    Example:

    ```python
    >>> from transformers import Phi3SmallConfig, Phi3SmallModel

    >>> # Initializing a Phi3Small configuration
    >>> configuration = Phi3SmallConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Phi3SmallModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "phi3small"
    keys_to_ignore_at_inference = ["past_key_values"]
    

    def __init__(
        self,
        # General information about the model
        vocab_size: int =100352,
        max_position_embeddings: int = 8192,
        # RoPE Related Parameters
        rope_embedding_base: float = 10**6,
        rope_position_scale: float = 1.0,
        rope_scaling: Optional[Dict[str, Union[float, List[float], int]]] = None,
        # General Model Parameters
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        # KV Shared Attention Configurations
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        # GEGELU Related Parameters
        hidden_act: str = "gegelu",
        gegelu_limit: float = 20.0,
        gegelu_pad_to_256: bool = True,
        ff_dim_multiplier: Optional[int] = None,
        ff_intermediate_size: Optional[int] = 14336,
        # Block Sparse Attention Parameters
        blocksparse_homo_head_pattern: bool = False,
        blocksparse_block_size: int = 64,
        blocksparse_num_local_blocks: int = 16,
        blocksparse_vert_stride: int = 8,
        blocksparse_triton_kernel_block_size: int = 64,
        # Frequency of block-sparsity
        dense_attention_every_n_layers: Optional[int] = 2,
        # Reegularization parameters
        embedding_dropout_prob: float =0.1,
        attention_dropout_prob: float = 0.0,
        ffn_dropout_prob: float = 0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        # MuP parameters
        mup_use_scaling: bool = True,
        mup_width_multiplier: bool = 8.0,
        mup_embedding_multiplier: bool = 10.0,
        mup_attn_multiplier: bool =1.0,
        use_cache=True,
        # The model does not have a bos token id
        # However, in order for some of the downstream libraries to not break
        # we set this to be the same as the eos_token_id
        bos_token_id: int = 100257,
        eos_token_id: int = 100257,
        reorder_and_upcast_attn=False,
        # Configuration to pad sequence length to a multiple of 64
        pad_sequence_to_multiple_of_64: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_embedding_base = rope_embedding_base
        self.rope_position_scale = rope_position_scale
        self.rope_scaling = rope_scaling
        self.hidden_size = hidden_size
        # QK Shared Attention
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # Block Sparse Attention Pattern
        self.blocksparse_homo_head_pattern = blocksparse_homo_head_pattern
        self.blocksparse_block_size = blocksparse_block_size
        self.blocksparse_num_local_blocks = blocksparse_num_local_blocks
        self.blocksparse_vert_stride = blocksparse_vert_stride
        self.blocksparse_triton_kernel_block_size = blocksparse_triton_kernel_block_size
        # Frequency of block sparsity
        self.dense_attention_every_n_layers = dense_attention_every_n_layers
        # Activation function
        self.hidden_act = hidden_act
        self.gegelu_limit = gegelu_limit
        self.gegelu_pad_to_256 = gegelu_pad_to_256
        self.ff_dim_multiplier = ff_dim_multiplier
        self.ff_intermediate_size = ff_intermediate_size
        if self.ff_dim_multiplier is None and self.ff_intermediate_size is None:
            raise ValueError(f"Cannot have both {self.ff_dim_multiplier} and {self.ff_intermediate_size} as None")
        if self.ff_dim_multiplier is not None and self.ff_intermediate_size is not None:
            raise ValueError(f"Cannot specify both {self.ff_dim_multiplier} and {self.ff_intermediate_size}.")
        # General regularization
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.ffn_dropout_prob = ffn_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        # MuP parameters
        self.mup_use_scaling = mup_use_scaling
        self.mup_width_multiplier = mup_width_multiplier
        self.mup_embedding_multiplier = mup_embedding_multiplier
        self.mup_attn_multiplier = mup_attn_multiplier
        self.use_cache = use_cache

        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.pad_sequence_to_multiple_of_64 = pad_sequence_to_multiple_of_64

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @cached_property
    def dummy_token_indices(self) -> List[int]:
        # Importing here to avoid circular imports
        from .tokenization_phi3_small import Phi3SmallTokenizer
        tokenizer = Phi3SmallTokenizer()
        return tokenizer.dummy_token_indices

    @property
    def intermediate_size(self) -> int:
        if self.ff_intermediate_size is not None:
            return self.ff_intermediate_size
        intermediate_size = (self.ff_dim_multiplier) * (self.hidden_size // 3) * 2
        if self.gegelu_pad_to_256:
            intermediate_size = next_mult(intermediate_size, 256)
        return intermediate_size