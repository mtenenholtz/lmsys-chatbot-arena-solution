"""
Orginally Taken verbatim from xformers library
https://github.com/facebookresearch/xformers/blob/bcb707576c6a80eaf850aa80e8643d3497ec2bc4/xformers/components/positional_embedding/rotary.py

The difference is that xformers seems to assume the inputs to be
(bs, head, seq_len, dim) while we assume (bs, seq_len, head, dim)

"""
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This implementation is inspired by GPT-NeoX https://github.com/EleutherAI/gpt-neox
# NOTE: Almost the same right now, moving parts to Triton is the next step

import math
from typing import List, Optional, Tuple, Dict, Union

import torch
import dataclasses
from transformers.utils import logging

from transformers import PretrainedConfig

is_dacite_available = False
try:
    import dacite
    is_dacite_available = True
except ImportError:
    pass

logger = logging.get_logger(__name__)

@dataclasses.dataclass
class LongRopeConfig(object):
    short_factor: List[float]
    long_factor: List[float]
    original_max_position_embeddings: int
    type: str = "longrope"
    short_mscale: float = -1
    long_mscale: float = -1


    def __post_init__(self):
        assert self.type in ("longrope", "su"), f"Invalid type {self.type} for LongRopeConfig. Expected longrope / su"


    @classmethod
    def from_dict(cls, config_dict: Dict[str, Union[float, List[float], int]]) -> "LongRopeConfig":
        if is_dacite_available:
            # Preferred since we can also type check the input
            return dacite.from_dict(data_class=cls, data=config_dict)
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name in config_dict:
                if field.init:
                    kwargs[field.name] = config_dict[field.name]
                else:
                    raise ValueError(f"Field {field.name} is not initiable")
            else:
                if field.default is dataclasses.MISSING:
                    raise ValueError(f"Field {field.name} is required")
        extra_keys = set(config_dict.keys()) - set(kwargs.keys())
        if len(extra_keys) > 0:
            for key in extra_keys:
                logger.error(f"Unrecognized key {key} in config_dict")
            raise ValueError(f"Unrecognized keys in config_dict")
        return cls(**kwargs)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)



@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin, seq_dimension: int):
    # NOTE: This could probably be moved to Triton

    if seq_dimension == 0:
        cos = cos[: x.shape[0], None, None, :]
        sin = sin[: x.shape[0], None, None, :]
    elif seq_dimension == 1:
        # Handle a possible sequence length mismatch in between q and k
        cos = cos[None, : x.shape[1], None, :]
        sin = sin[None, : x.shape[1], None, :]
    elif seq_dimension == 2:
        cos = cos[None, None, : x.shape[2], :]
        sin = sin[None, None, : x.shape[2], :]

    return (x * cos) + (rotate_half(x) * sin)



class RotaryEmbedding(torch.nn.Module):
    """
    Adapted from the xformers library

    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis

    # Arguments
    :param dim_mode: head dimention
    :param max_seq_len:
    :param default_seq_dimension: which dim is the sequence length
    :param dtype: cos/sin dtype
    :param use_fused_kernel: if to use customized fused kernel.
        Note: if used, q, k will be modified inplace. Ok for both forward & backward.
    """

    def __init__(
        self,
        dim_model: int,
        *,
        max_seq_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        base=10000,
        position_scale=1,
        device: Optional[torch.device] = None,
        longrope_config: Optional[LongRopeConfig] = None,
    ):
        super().__init__()
        self.base = base
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        self.longrope_config = longrope_config

        if self.is_longrope:
            # Keep the maximum range vector, and slice from it as needed
            self.register_buffer(
                "range_vector",
                torch.arange(max_seq_len, device=device, dtype=torch.float32),
                persistent=False
            )
            self.register_buffer(
                "short_factors",
                torch.tensor(self.longrope_config.short_factor, dtype=torch.float32),
                persistent=False
            )
            self.register_buffer(
                "long_factors",
                torch.tensor(self.longrope_config.long_factor, dtype=torch.float32),
                persistent=False
            )
        else:
            # Generate and save the inverse frequency buffer (non trainable)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim_model, 2).float().to(device) / self.dim_model))
            self.register_buffer("inv_freq", inv_freq)

        self.position_scale = position_scale
        
        if not self.is_longrope:
            dtype = dtype or torch.get_default_dtype()
            self._set_cos_sin_cache(
                seq_len=max_seq_len,
                device=self.inv_freq.device,
                dtype=dtype,
            )
    @property
    def is_longrope(self):
        return self.longrope_config is not None

    @property
    def original_max_seq_len(self):
        if self.longrope_config is not None:
            return self.longrope_config.original_max_position_embeddings
        logger.warning_once(
            (
                "``original_max_seq_len'' is being accessed, but longrope_config has not been set. "
                "Please only do this if you are sure about the context."
            )
        )
        return self.max_seq_len

    def get_range_vector(self, seq_len: int, device: torch.device):
        if self.is_longrope:
            assert seq_len < self.range_vector.shape[0], f"Found seq_len {seq_len} greater than max_seq_len {self.range_vector.shape[0]}"
            if self.range_vector.device != device:
                self.range_vector = self.range_vector.to(device)
            return self.range_vector[:seq_len]
        return torch.arange(seq_len, device=device, dtype=torch.float32)


    def _calc_mscale(self, scale: torch.Tensor) -> torch.Tensor:
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_seq_len))

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        dtype = dtype or torch.get_default_dtype()
        self.max_seq_len_cached = seq_len
        t = (torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32) * self.position_scale).type_as(self.inv_freq)
        device_type = device.type if device is not None else "cpu"
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # shape: (seq_len, dim_model // 2)
            freqs = torch.outer(t, self.inv_freq)
            # shape: (seq_len, dim_model)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def forward(
        self, q: torch.Tensor,
        k: torch.Tensor,
        seq_dimension: int = 1,
        seqlen_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q, k does not include `seqlen_offset`
        q: Either (bs, seq_len, num_heads, head_dim) or (seq_len, bs, num_heads, head_dim)
        k: Either (bs, seq_len, num_heads, head_dim) or (seq_len, bs, num_heads, head_dim)
        """
        if seq_dimension < 0:
            seq_dimension = k.ndim + seq_dimension
        assert seq_dimension in (0, 1, 2)
        seq_len = k.shape[seq_dimension] + seqlen_offset

        if self.is_longrope:
            if seq_len > self.original_max_seq_len:
                t = self.get_range_vector(seq_len, device=q.device)
                rescale_factors = self.long_factors.to(q.device)
                long_mscale = self.longrope_config.long_mscale
                mscale = long_mscale if long_mscale > 0 else self._calc_mscale(self.max_seq_len / self.original_max_seq_len)
            else:
                t = self.get_range_vector(self.original_max_seq_len, device=q.device)
                rescale_factors = self.short_factors.to(q.device)
                short_mscale = self.longrope_config.short_mscale
                mscale = short_mscale if short_mscale > 0 else 1.0
            assert rescale_factors.shape == (self.dim_model // 2, ), (
                f"misaligned shape for LongRoPE rescale factors:\n"
                f"\tExpected {(self.dim_model // 2, )}, got {rescale_factors.shape}."
            )
            inv_freq = 1.0 / (rescale_factors * (self.base ** (torch.arange(0, self.dim_model, 2).float().to(q.device) / self.dim_model)))
            device_type = q.device.type if q.device is not None else "cpu"
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * mscale
                sin = emb.sin() * mscale
            cos_cached = cos.to(q.dtype)
            sin_cached = sin.to(q.dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(
                    seq_len=seq_len,
                    device=k.device,
                    dtype=k.dtype,
                )
            cos_cached = self.cos_cached
            sin_cached = self.sin_cached
        return (
            apply_rotary_pos_emb(
                q, cos_cached[seqlen_offset:seq_len], sin_cached[seqlen_offset:seq_len], seq_dimension=seq_dimension
            ).to(q.dtype),
            apply_rotary_pos_emb(
                k, cos_cached[seqlen_offset:seq_len], sin_cached[seqlen_offset:seq_len], seq_dimension=seq_dimension
            ).to(q.dtype),
        )

    @classmethod
    def from_config(cls, config: PretrainedConfig) -> "RotaryEmbedding":
        kwargs = dict(
            dim_model=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_embedding_base,
            position_scale=config.rope_position_scale,
        )
        if config.rope_scaling is not None:
            kwargs["longrope_config"] = LongRopeConfig.from_dict(config.rope_scaling)
        return cls(**kwargs)