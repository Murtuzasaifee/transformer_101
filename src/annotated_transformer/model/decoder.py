"""Decoder stack: N identical layers, each with self-attn, cross-attn, and feed-forward."""

import torch
import torch.nn as nn

from annotated_transformer.model.sublayer import LayerNorm, SublayerConnection, clones


class DecoderLayer(nn.Module):
    """One decoder layer: masked self-attention, encoder-decoder cross-attention, feed-forward."""

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = memory
        # 1) Masked self-attention over the target sequence so far. tgt_mask
        #    combines padding with a causal mask (see Batch.make_std_mask) so
        #    position i can only look at positions <= i -- the decoder must
        #    not "cheat" by seeing future tokens it's supposed to predict.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) Cross-attention: queries come from the decoder (x), but keys/
        #    values come from the encoder's output (m = memory). This is how
        #    the decoder "reads" the source sentence while generating the
        #    target -- e.g. attending to the German words relevant to the
        #    English word currently being produced.
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 3) Position-wise feed-forward, same as in the encoder.
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """Stack of N DecoderLayers with a final layer norm."""

    def __init__(self, layer: DecoderLayer, n: int):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
