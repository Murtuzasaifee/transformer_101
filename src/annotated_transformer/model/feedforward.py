"""Position-wise feed-forward network (Section 3.3)."""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Two linear transforms with a ReLU in between, applied per-position: FFN(x) = W2(relu(W1 x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model). w_1 expands to d_ff, ReLU adds the
        # non-linearity (without it, two stacked linear layers would collapse
        # into a single linear layer), w_2 projects back down to d_model so
        # the output can be added back via the residual connection. This runs
        # identically and independently on every position (hence
        # "position-wise") -- there's no mixing across seq_len here, that's
        # entirely the attention sublayer's job.
        return self.w_2(self.dropout(self.w_1(x).relu()))
