"""Vocabulary: build from the Multi30k corpus (or load a cached copy) and index tokens."""

import logging
from collections import Counter
from os.path import exists
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
import torch
from datasets import load_dataset

from annotated_transformer.data.tokenizer import tokenize

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<s>", "</s>", "<blank>", "<unk>"]


class Vocab:
    """Minimal string<->index vocabulary (a drop-in replacement for torchtext.vocab.Vocab)."""

    def __init__(self, stoi: Dict[str, int], itos: Dict[int, str], default_idx: int = 3):
        self.stoi = stoi
        self.itos = itos
        self.default_idx = default_idx

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_idx)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.default_idx) for t in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    def get_itos(self) -> Dict[int, str]:
        return self.itos

    def get_stoi(self) -> Dict[str, int]:
        return self.stoi

    def set_default_index(self, idx: int) -> None:
        self.default_idx = idx


def build_vocabulary(spacy_de: spacy.Language, spacy_en: spacy.Language) -> Tuple[Vocab, Vocab]:
    """Build source (German) and target (English) vocabularies from the Multi30k corpus."""

    def tokenize_de(text: str) -> List[str]:
        return tokenize(text, spacy_de)

    def tokenize_en(text: str) -> List[str]:
        return tokenize(text, spacy_en)

    logger.info("Building German vocabulary...")
    ds = load_dataset("bentrevett/multi30k", trust_remote_code=False)
    counter_de: Counter = Counter()
    counter_en: Counter = Counter()
    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            counter_de.update(tokenize_de(ex["de"]))
            counter_en.update(tokenize_en(ex["en"]))

    stoi_de = {w: i for i, w in enumerate(SPECIAL_TOKENS)}
    for w, _ in counter_de.most_common():
        if w not in stoi_de:
            stoi_de[w] = len(stoi_de)
    itos_de = {i: w for w, i in stoi_de.items()}

    logger.info("Building English vocabulary...")
    stoi_en = {w: i for i, w in enumerate(SPECIAL_TOKENS)}
    for w, _ in counter_en.most_common():
        if w not in stoi_en:
            stoi_en[w] = len(stoi_en)
    itos_en = {i: w for w, i in stoi_en.items()}

    return Vocab(stoi_de, itos_de), Vocab(stoi_en, itos_en)


def load_vocab(
    spacy_de: spacy.Language, spacy_en: spacy.Language, cache_path: str = "vocab.pt"
) -> Tuple[Vocab, Vocab]:
    """Load a cached vocab pair from disk, building and caching it if absent."""
    if not exists(cache_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save((vocab_src, vocab_tgt), cache_path)
    else:
        vocab_src, vocab_tgt = torch.load(cache_path, map_location="cpu", weights_only=False)
    logger.info(
        "Vocabulary sizes -- src (de): %d, tgt (en): %d", len(vocab_src), len(vocab_tgt)
    )
    return vocab_src, vocab_tgt
