#!/usr/bin/env python
"""Train the Transformer on Multi30k (German -> English).

Meant to run on a GPU server. Uses a single GPU by default; set
`training.distributed: true` in the config (or pass --distributed) to train
across all visible GPUs with DistributedDataParallel.

Usage:
    uv run scripts/train.py --config configs/multi30k.yaml
    uv run scripts/train.py --config configs/multi30k.yaml --distributed
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annotated_transformer.config import load_config  # noqa: E402
from annotated_transformer.data.tokenizer import load_spacy_tokenizers  # noqa: E402
from annotated_transformer.data.vocab import load_vocab  # noqa: E402
from annotated_transformer.training.trainer import train_model  # noqa: E402
from annotated_transformer.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML experiment config")
    parser.add_argument(
        "--distributed", action="store_true", help="Override config: train with DDP across all GPUs"
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    if args.distributed:
        config.training.distributed = True

    logger.info("Experiment: %s", config.experiment_name)
    logger.info("Model config: %s", config.model.model_dump())
    logger.info("Training config: %s", config.training.model_dump())

    logger.info("Loading spaCy tokenizers...")
    spacy_de, spacy_en = load_spacy_tokenizers()

    logger.info("Loading vocabulary (cache: %s)...", config.data.vocab_cache_path)
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en, cache_path=config.data.vocab_cache_path)

    logger.info("Starting training...")
    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    logger.info("Training complete. Checkpoints saved under %s", config.training.checkpoint_dir)


if __name__ == "__main__":
    main()
