"""Top-level EncoderDecoder model and the factory that wires all sub-modules together."""

import copy

import torch
import torch.nn as nn

from annotated_transformer.model.attention import MultiHeadedAttention
from annotated_transformer.model.decoder import Decoder, DecoderLayer
from annotated_transformer.model.embeddings import Embeddings, PositionalEncoding
from annotated_transformer.model.encoder import Encoder, EncoderLayer
from annotated_transformer.model.feedforward import PositionwiseFeedForward
from annotated_transformer.model.generator import Generator


class EncoderDecoder(nn.Module):
    """Standard encoder-decoder architecture. Base for this and most seq2seq models."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: Generator,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode src once, then decode tgt against the resulting memory."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def build_transformer_model(
    src_vocab: int,
    tgt_vocab: int,
    n_layers: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    n_heads: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """Construct a full EncoderDecoder Transformer from hyperparameters.

    Each attention/feed-forward/positional-encoding submodule is instantiated once
    and deep-copied per layer so layers don't share weights.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Glorot / fan_avg initialization, as in the reference implementation.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
