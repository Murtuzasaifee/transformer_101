"""Scaled dot-product attention and multi-head attention (Section 3.2 of the paper)."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from annotated_transformer.model.sublayer import clones


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    Args:
        query, key, value: shape (batch, heads, seq_len, d_k)
        mask: broadcastable boolean/int tensor; positions where mask == 0 are
            blocked (set to -inf before softmax)
        dropout: optional dropout applied to the attention weights

    Returns:
        (output, attention_weights)
    """
    d_k = query.size(-1)

    # QK^T: for every query position, get a raw similarity score against every
    # key position. key.transpose(-2, -1) flips the last two dims so the
    # matmul contracts over d_k, giving shape (..., seq_len_q, seq_len_k).
    # Dividing by sqrt(d_k) keeps the dot products from growing too large as
    # d_k increases, which would otherwise push softmax into a near-one-hot,
    # near-zero-gradient regime.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Wherever mask == 0 (padding, or "future" positions in the decoder),
        # force the score to a huge negative number so softmax below sends its
        # probability to ~0 -- effectively "cannot attend to that position".
        scores = scores.masked_fill(mask == 0, -1e9)

    # Turn raw scores into a probability distribution over key positions
    # (each query's attention weights sum to 1 across the last dimension).
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Weighted sum of the value vectors, weighted by how much each query
    # attends to each key -- this is the actual "attention output".
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Splits d_model into `h` heads, applies attention per-head, then recombines."""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by the number of heads"
        self.d_k = d_model // h  # we assume d_v == d_k
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # Q, K, V, output projections
        self.attn: Optional[torch.Tensor] = None  # last attention weights, kept for inspection/viz
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            # mask starts as (batch, seq_len_q, seq_len_k); insert a dim at
            # position 1 so it broadcasts across the head dimension we're
            # about to introduce below -> (batch, 1, seq_len_q, seq_len_k).
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Linearly project Q, K, V (each: batch, seq_len, d_model), then
        #    reshape d_model -> (h, d_k) and move the head dim next to batch:
        #    .view(...)   splits the last dim into h heads of size d_k each
        #    .transpose() swaps seq_len and h so shape becomes
        #                 (batch, h, seq_len, d_k) -- attention below then
        #                 treats each head as an independent batch element,
        #                 so all heads are computed in one parallel matmul.
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Run scaled dot-product attention independently per head (the
        #    head dimension just rides along as an extra batch dimension).
        x, self.attn = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) Undo step 1's reshape: move heads back next to seq_len, then
        #    flatten (h, d_k) back into a single d_model-sized vector per
        #    token. `.contiguous()` is required because `.transpose()` only
        #    changes strides, and `.view()` needs contiguous memory.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query, key, value
        # Final linear layer mixes information across heads before returning.
        return self.linears[-1](x)
