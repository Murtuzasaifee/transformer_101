"""Core Transformer architecture, decomposed into independently replaceable modules.

Each sub-layer (attention, feed-forward, embeddings, encoder/decoder stacks) lives
in its own file so any single piece can be swapped out (e.g. a different attention
kernel or positional encoding scheme) without touching the rest of the model.
"""

from annotated_transformer.model.attention import MultiHeadedAttention, scaled_dot_product_attention
from annotated_transformer.model.embeddings import Embeddings, PositionalEncoding
from annotated_transformer.model.encoder import Encoder, EncoderLayer
from annotated_transformer.model.decoder import Decoder, DecoderLayer
from annotated_transformer.model.feedforward import PositionwiseFeedForward
from annotated_transformer.model.generator import Generator
from annotated_transformer.model.mask import subsequent_mask
from annotated_transformer.model.sublayer import LayerNorm, SublayerConnection, clones
from annotated_transformer.model.transformer import EncoderDecoder, build_transformer_model

__all__ = [
    "MultiHeadedAttention",
    "scaled_dot_product_attention",
    "Embeddings",
    "PositionalEncoding",
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "PositionwiseFeedForward",
    "Generator",
    "subsequent_mask",
    "LayerNorm",
    "SublayerConnection",
    "clones",
    "EncoderDecoder",
    "build_transformer_model",
]
