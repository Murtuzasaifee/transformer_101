"""Token embeddings and sinusoidal positional encoding (Section 3.4 / 3.5)."""

import math

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """Learned token embedding, scaled by sqrt(d_model) as in the paper."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        # lut = "lookup table": maps each token id to a learned d_model-sized
        # vector. Shape in -> out: (batch, seq_len) -> (batch, seq_len, d_model).
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The sqrt(d_model) scale-up is from the paper: it makes the
        # embedding magnitudes comparable to the positional encodings added
        # next (embeddings would otherwise be relatively small), so neither
        # signal drowns out the other once they're summed.
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding, added to token embeddings.

    Self-attention has no built-in notion of word order (unlike an RNN, which
    processes tokens one after another) -- every position attends to every
    other position symmetrically. To let the model tell "the cat sat" apart
    from "sat the cat", we inject a unique, deterministic pattern per position
    directly into the embeddings, using sine/cosine waves of different
    frequencies (Section 3.5 of the paper).
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute one encoding vector per position, up to max_len, so the
        # forward pass is just a lookup/slice instead of recomputing sin/cos
        # every call. Shape: (max_len, d_model).
        pe = torch.zeros(max_len, d_model)
        # Column vector of positions [0, 1, 2, ..., max_len-1], shape (max_len, 1)
        # so it broadcasts against div_term (shape (d_model/2,)) below.
        position = torch.arange(0, max_len).unsqueeze(1)
        # One frequency per pair of dimensions, spaced geometrically from 1
        # down to ~1/10000 -- low dimensions oscillate fast (encode fine-grained
        # position), high dimensions oscillate slowly (encode coarse position).
        # Computed via exp(log(...)) instead of a direct power for numerical
        # stability with large d_model.
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # Even dimensions get sin, odd dimensions get cos of the same
        # frequency -- this pairing is what lets the model later compute
        # relative offsets via a fixed linear transform (see paper, Sec 3.5).
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # add a batch dim so it broadcasts: (1, max_len, d_model)
        # register_buffer: not a learnable nn.Parameter (no gradient), but
        # still moves with the module when you call .to(device)/.cuda(), and
        # gets saved/restored by state_dict() like a parameter would.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice out only as many positions as the current sequence needs, and
        # add them directly to the token embeddings (same shape, so this is
        # elementwise addition -- position info and content info now share
        # the same vector). requires_grad_(False) is redundant here (buffers
        # already don't track gradients) but documents intent explicitly.
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
