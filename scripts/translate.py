#!/usr/bin/env python
"""Load a trained checkpoint and print greedy-decoded translations for validation examples.

Usage:
    uv run scripts/translate.py --config configs/multi30k.yaml \
        --checkpoint checkpoints/multi30k_model_final.pt --n-examples 10
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from annotated_transformer.config import load_config  # noqa: E402
from annotated_transformer.data.dataset import create_dataloaders  # noqa: E402
from annotated_transformer.data.tokenizer import load_spacy_tokenizers  # noqa: E402
from annotated_transformer.data.vocab import load_vocab  # noqa: E402
from annotated_transformer.inference.evaluate import check_outputs  # noqa: E402
from annotated_transformer.model.transformer import build_transformer_model  # noqa: E402
from annotated_transformer.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML experiment config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model .pt checkpoint")
    parser.add_argument("--n-examples", type=int, default=10)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    device = torch.device("cpu")

    logger.info("Loading spaCy tokenizers and vocabulary...")
    spacy_de, spacy_en = load_spacy_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, cache_path=config.data.vocab_cache_path)

    logger.info("Building model and loading weights from %s...", args.checkpoint)
    model = build_transformer_model(
        len(vocab_src),
        len(vocab_tgt),
        n_layers=config.model.n_layers,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    _, valid_dataloader = create_dataloaders(
        device, vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=1, is_distributed=False
    )

    check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=args.n_examples)


if __name__ == "__main__":
    main()
