"""Encoder stack: N identical layers, each with self-attention + feed-forward sublayers."""

import torch
import torch.nn as nn

from annotated_transformer.model.sublayer import LayerNorm, SublayerConnection, clones


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention sublayer followed by feed-forward sublayer."""

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # sublayer[0] wraps self-attention with LayerNorm + residual (see
        # SublayerConnection). We pass a lambda because SublayerConnection
        # needs to call the sublayer on the *normalized* x, not the raw x --
        # `x, x, x` means query/key/value all come from the same sequence,
        # which is what makes this "self" attention (every token looks at
        # every other token in the same sentence).
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # sublayer[1] wraps the position-wise feed-forward network the same way.
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Stack of N EncoderLayers with a final layer norm."""

    def __init__(self, layer: EncoderLayer, n: int):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Feed x through each layer in sequence -- layer i's output becomes
        # layer i+1's input, progressively building richer representations.
        for layer in self.layers:
            x = layer(x, mask)
        # Final norm cleans up the output of the last residual addition
        # before it's used as "memory" for the decoder's cross-attention.
        return self.norm(x)
