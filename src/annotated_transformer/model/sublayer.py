"""Shared building blocks: layer cloning, LayerNorm, and residual sublayer wrapping."""

import copy

import torch
import torch.nn as nn


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Produce N deep-copied, independently-parameterized instances of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale (a_2) and shift (b_2).

    Unlike BatchNorm (which normalizes across the batch), LayerNorm normalizes
    across the *feature* dimension of a single token -- so it works the same
    whether batch size is 1 or 1000, which is why Transformers use it.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        # a_2 (scale) and b_2 (shift) are learned so the network can undo the
        # normalization if that turns out to help -- they start as identity
        # (multiply by 1, add 0) and are nudged during training.
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  # tiny constant to avoid divide-by-zero if std collapses to 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean/std per token (last dim = feature/d_model dim), not across
        # the batch or sequence -- keepdim=True so the result still broadcasts
        # against x for the subtraction/division below.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # Standardize to zero mean / unit variance, then apply the learned
        # scale + shift: this is the standard LayerNorm formula.
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection followed by layer norm (pre-norm variant).

    Note: norm is applied first (pre-norm) rather than after the residual add,
    which differs slightly from the original paper's diagram but trains more
    stably and matches the reference "Annotated Transformer" implementation.
    """

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same output size.

        `sublayer` is a callable (e.g. self-attention or feed-forward) passed
        in by the caller. The flow is: normalize x -> run the sublayer on the
        normalized version -> dropout the result -> add it back to the
        *original*, un-normalized x. That `x +` is the residual/skip
        connection: it gives gradients a direct path back through every layer,
        which is what makes very deep Transformer stacks trainable at all.
        """
        return x + self.dropout(sublayer(self.norm(x)))
