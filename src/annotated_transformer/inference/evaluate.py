"""Qualitative evaluation: decode a handful of validation examples and print translations."""

import logging
from typing import List, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from annotated_transformer.data.batch import Batch
from annotated_transformer.data.vocab import Vocab
from annotated_transformer.inference.decode import greedy_decode

logger = logging.getLogger(__name__)


def check_outputs(
    valid_dataloader: DataLoader,
    model: nn.Module,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    n_examples: int = 15,
    pad_idx: int = 2,
    eos_string: str = "</s>",
) -> List[Tuple]:
    """Greedy-decode `n_examples` validation batches and log source/target/prediction."""
    results = [()] * n_examples
    data_iter = iter(valid_dataloader)

    for idx in range(n_examples):
        b = next(data_iter)
        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [vocab_src.get_itos()[int(x)] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[int(x)] for x in rb.tgt[0] if x != pad_idx]

        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(vocab_tgt.get_itos()[int(x)] for x in model_out if x != pad_idx).split(
                eos_string, 1
            )[0]
            + eos_string
        )

        logger.info("Example %d ========", idx)
        logger.info("Source Text (Input)        : %s", " ".join(src_tokens))
        logger.info("Target Text (Ground Truth) : %s", " ".join(tgt_tokens))
        logger.info("Model Output               : %s", model_txt)

        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)

    return results
