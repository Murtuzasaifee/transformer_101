"""spaCy-based tokenizers for the German (source) / English (target) language pair."""

import logging
import os
from typing import List, Tuple

import spacy

logger = logging.getLogger(__name__)


def load_spacy_tokenizers() -> Tuple[spacy.Language, spacy.Language]:
    """Load German + English spaCy pipelines, downloading the small models on first use."""
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        logger.info("de_core_news_sm not found locally, downloading...")
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        logger.info("en_core_web_sm not found locally, downloading...")
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text: str, tokenizer: spacy.Language) -> List[str]:
    """Tokenize raw text into a list of surface-form tokens using a spaCy pipeline's tokenizer."""
    return [tok.text for tok in tokenizer.tokenizer(text)]
