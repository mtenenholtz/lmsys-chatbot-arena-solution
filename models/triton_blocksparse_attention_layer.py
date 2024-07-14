import math
from typing import Optional, Tuple, TypeVar
import torch.nn as nn
import torch
import triton

from functools import lru_cache


from .triton_flash_blocksparse_attn import get_local_strided_sparse_attention_op, _get_sparse_attn_mask, blocksparse_flash_attn_padded_fwd, blocksparse_flash_attn_varlen_fwd


Layout = Tuple[torch.LongTensor, torch.LongTensor]


def create_sparse_attn_mask(
    n_heads: int,
    max_seq_len: int,
    max_seq_len_k: int,
    dtype: torch.dtype,
    device: torch.device,
    BLOCK: int,
    local_blocks: int,
    vert_stride: int,
    homo_head: bool,
    return_dense: bool
) -> Tuple[Layout, torch.Tensor, Optional[torch.Tensor]]:
    layout, block_sparse_pattern, _ = _get_sparse_attn_mask(
        n_heads=n_heads,
        q_len=max_seq_len,
        N_CTX=max_seq_len_k,
        dtype=dtype,
        device=device,
        BLOCK=BLOCK,
        local_blocks=local_blocks,
        vert_stride=vert_stride,
        homo_head=homo_head,
        return_dense=return_dense
    )
    return layout, block_sparse_pattern


class BlockSparseAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        max_seq_len: int,
        sparse_block_size: int,
        local_blocks: int,
        vert_stride: int,
        kernel_block_size: Optional[int] = None,
        homo_head: bool = False,
        active_head_range: Optional[Tuple[int]] = None
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.sparse_block_size = sparse_block_size
        self.kernel_block_size = kernel_block_size or sparse_block_size
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.homo_head = homo_head
        self.active_head_range = active_head_range

        # Internal Parameters used by the layer
        self._sparse_block_mask = None
        self._sparse_layout = None
        self._dtype = None
        self._device = None

        # TODO(bapatra): Ideally, I'd want to keep all the code for
        # forward to be handled here, and not branch for training and inference.
        # However, that refactor would need a lot of testing. For now, using the
        # training op as is, and will refactor again later.
    
    def prune_blocksparse_layout_to_heads(self, h_start: int, h_end: int) -> None:
        self._sparse_block_mask = self._sparse_block_mask[h_start: h_end]
        self._sparse_layout[0] = self._sparse_layout[0][h_start: h_end]
        self._sparse_layout[1] = self._sparse_layout[1][h_start: h_end]
    
    def _initialize_internals(
        self,
        dtype: torch.dtype,
        device: torch.device
    ) -> None:
        self._dtype, self._device = dtype, device
        self._sparse_layout, self._sparse_block_mask = create_sparse_attn_mask(
            n_heads=self.n_heads,
            max_seq_len=self.max_seq_len,
            max_seq_len_k=self.max_seq_len,
            dtype=dtype,
            device=device,
            BLOCK=self.sparse_block_size,
            local_blocks=self.local_blocks,
            vert_stride=self.vert_stride,
            homo_head=self.homo_head,
            return_dense=False,
        )
        if (not self.homo_head) and (self.active_head_range is not None):
            assert len(self.active_head_range) == 2, "\"active_head_range\" should be a tuple of start/end index of the heads."
            h_start, h_end = self.active_head_range
            self.prune_blocksparse_layout_to_heads(h_start=h_start, h_end=h_end)

        assert self.sparse_block_size % self.kernel_block_size == 0,  f"The sparse block size must be a multiple of {self.kernel_block_size}. Found {self.sparse_block_size}."
        assert self.kernel_block_size >=16 and math.log2(self.kernel_block_size) % 1 == 0, f"block_size must be power of 2 and at least 16, but {self.kernel_block_size} is given"
        if self.sparse_block_size // self.kernel_block_size > 1:
            _mul = self.sparse_block_size // self.kernel_block_size
            # need to consider if block_m and block_n are different
            self._sparse_block_mask = torch.kron(self._sparse_block_mask, self._sparse_block_mask.new_ones(_mul, _mul))
            num_sparse_blocks = self._sparse_block_mask.size(-1)
            block_causal_mask = torch.arange(0, num_sparse_blocks)[:, None] >= torch.arange(0, num_sparse_blocks)[None]
            self._sparse_block_mask *= block_causal_mask.type_as(self._sparse_block_mask)


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
        *,
        # Arguments Related to Block Attention Inference
        left_paddings: Optional[torch.LongTensor] = None,
        seqlens: Optional[torch.LongTensor] = None,
        # Arguements Related to Variable Length Inference
        cu_seqlens_k: Optional[torch.LongTensor] = None,
        cu_seqlens_q: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        if left_paddings is None and seqlens is None and cu_seqlens_k is None and cu_seqlens_q is None:
            blocksparse_op = get_local_strided_sparse_attention_op(
                n_heads=self.n_heads,
                max_seq_len=self.max_seq_len,
                sparse_block_size=self.sparse_block_size,
                kernel_block_size=self.kernel_block_size,
                local_blocks=self.local_blocks,
                vert_stride=self.vert_stride,
                homo_head=self.homo_head,
                device=q.device,
                inference=not self.training
            )
            return blocksparse_op(q, k, v, sm_scale)

        assert not torch.is_grad_enabled(), "Variable Length Inference / Batched inference is not supported during training. Please run it in a torch.no_grad() context"
        # First set internals if they have not been set
        if self._sparse_block_mask is None or (self._dtype != q.dtype) or (self._device != q.device):
            self._initialize_internals(dtype=q.dtype, device=q.device)
        
        if k.dim() == 3:
            assert cu_seqlens_k is not None
            return blocksparse_flash_attn_varlen_fwd(
                q=q,
                k=k,
                v=v,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_q=cu_seqlens_q,
                sm_scale=sm_scale,
                sparse_layout=self._sparse_layout,
                block_size=self.kernel_block_size,
                max_seqlen=self.max_seq_len,
            )
        if k.dim() == 4:
            assert not (left_paddings is None and seqlens is None), "Either left_paddings or seqlens must be provided for batched inference."
            return blocksparse_flash_attn_padded_fwd(
                q=q,
                k=k,
                v=v,
                sm_scale=sm_scale,
                sparse_layout=self._sparse_layout,
                left_paddings=left_paddings,
                seqlens=seqlens,
                block_size=self.kernel_block_size,
                max_seqlen=self.max_seq_len,
            )
        raise ValueError('q/k/v must be either 3 dim for variable-length input or 4 dim for fixed-length.')