"""Greedy autoregressive decoding."""

import torch
import torch.nn as nn

from annotated_transformer.model.mask import subsequent_mask


def greedy_decode(
    model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int
) -> torch.Tensor:
    """Encode `src` once, then greedily pick the argmax token at each decoding step."""
    # The encoder only needs to run once -- "memory" (its output) is reused
    # for every decoding step below, unlike the decoder which must re-run
    # each step since its input sequence keeps growing.
    memory = model.encode(src, src_mask)

    # Start the output sequence with just the <s> (start) token.
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    # Autoregressive generation: at each step, feed everything generated so
    # far back into the decoder, take the prediction for the *last* position
    # only (that's the model's guess for the next word), append it, repeat.
    # There's no early-stop on </s> here -- it always runs to max_len and the
    # caller (e.g. check_outputs) truncates at the first </s> afterwards.
    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        # out[:, -1] = decoder's hidden state for the most recently generated
        # position -- that's the one that predicts the *next* token.
        prob = model.generator(out[:, -1])
        # Greedy = always take the single highest-probability token (no beam
        # search, no sampling) -- simple, deterministic, but not necessarily
        # the globally best translation.
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
