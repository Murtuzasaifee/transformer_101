"""Batch: pairs a src/tgt tensor pair with the masks the model needs at train time."""

import torch

from annotated_transformer.model.mask import subsequent_mask


class Batch:
    """Holds one batch of source/target token ids plus their attention masks.

    The target is split into `tgt` (decoder input, shifted left) and `tgt_y`
    (the labels to predict, shifted right) -- standard teacher-forcing setup.
    """

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor = None, pad: int = 2):
        self.src = src
        # True wherever src is a real token, False wherever it's padding --
        # unsqueeze(-2) adds a query-position dim so this broadcasts against
        # attention scores of shape (batch, heads, seq_len_q, seq_len_k).
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            # Teacher forcing: the decoder is trained to predict each token
            # given everything before it, using the *ground-truth* previous
            # tokens as input (not its own possibly-wrong predictions).
            # `tgt` is fed into the decoder input side; `tgt_y` is what the
            # loss compares the decoder's predictions against. Slicing off
            # the last token from tgt and the first token from tgt_y shifts
            # them by one position relative to each other, e.g. for
            # tgt = [<s>, "a", "cat", "sat", </s>]:
            #   self.tgt   = [<s>, "a", "cat", "sat"]   (decoder input)
            #   self.tgt_y = ["a", "cat", "sat", </s>]  (labels to predict)
            # so at position i, the decoder input is token i and the label
            # is token i+1 -- "predict the next word".
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # Count of real (non-padding) label tokens in this batch -- used
            # to normalize the loss so batches with more/fewer real tokens
            # are weighted fairly (see SimpleLossCompute).
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        """Combine the padding mask with the causal (subsequent-position) mask."""
        # A position is only attendable if it's (a) a real token, not padding,
        # AND (b) not in the future relative to the current query position.
        # `&` combines both conditions elementwise (broadcasting the causal
        # mask's batch dim of 1 against the padding mask's real batch size).
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
