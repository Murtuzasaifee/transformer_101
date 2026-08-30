"""Masking utilities for autoregressive decoding."""

import torch


def subsequent_mask(size: int) -> torch.Tensor:
    """Build a causal (lower-triangular) mask so position i cannot attend to j > i.

    Returns a boolean tensor of shape (1, size, size) where True means "allowed
    to attend".

    Example for size=4 (rows = query position, columns = key position,
    True/1 = allowed, False/0 = blocked):

        [[1, 0, 0, 0],   # position 0 can only see itself
         [1, 1, 0, 0],   # position 1 can see 0 and 1
         [1, 1, 1, 0],   # position 2 can see 0, 1, 2
         [1, 1, 1, 1]]   # position 3 can see everything so far

    This is what forces the decoder to generate autoregressively -- token i's
    prediction can never peek at tokens after it.
    """
    attn_shape = (1, size, size)
    # torch.triu(..., diagonal=1) keeps only the *strictly upper* triangle
    # (everything above the main diagonal) and zeros out the rest -- that
    # upper triangle is exactly the "future" positions we want to block.
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)
    # Flip it: `mask == 0` turns "1 = future/blocked" into "True = allowed",
    # matching the convention scaled_dot_product_attention expects (it blocks
    # wherever the mask is 0/False).
    return mask == 0
