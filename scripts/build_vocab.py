#!/usr/bin/env python
"""Build (or rebuild) the source/target vocabulary cache from the Multi30k corpus.

Usage:
    uv run scripts/build_vocab.py --config configs/multi30k.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annotated_transformer.config import load_config  # noqa: E402
from annotated_transformer.data.tokenizer import load_spacy_tokenizers  # noqa: E402
from annotated_transformer.data.vocab import build_vocabulary  # noqa: E402
from annotated_transformer.utils import setup_logging  # noqa: E402

import torch  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML experiment config")
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the cache file already exists"
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    cache_path = Path(config.data.vocab_cache_path)

    if cache_path.exists() and not args.force:
        logger.info("Vocab cache already exists at %s (use --force to rebuild)", cache_path)
        return

    logger.info("Loading spaCy tokenizers...")
    spacy_de, spacy_en = load_spacy_tokenizers()

    logger.info("Building vocabulary from %s...", config.data.dataset_name)
    vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((vocab_src, vocab_tgt), cache_path)
    logger.info(
        "Saved vocab to %s (src=%d tokens, tgt=%d tokens)",
        cache_path,
        len(vocab_src),
        len(vocab_tgt),
    )


if __name__ == "__main__":
    main()
