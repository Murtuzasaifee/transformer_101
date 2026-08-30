"""Final projection + softmax that turns decoder output into a vocabulary distribution."""

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """Linear projection to vocab size followed by log-softmax."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) decoder output -> proj gives
        # (batch, seq_len, vocab_size) raw scores ("logits") per token.
        # log_softmax converts those into log-probabilities that sum to 1
        # per token (in probability space); using log directly (rather than
        # softmax then log) is numerically more stable and pairs naturally
        # with KLDivLoss / NLLLoss, which both expect log-probabilities.
        return log_softmax(self.proj(x), dim=-1)
