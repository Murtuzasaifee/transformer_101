"""Multi30k dataset wrapper, collation, and DataLoader construction."""

from typing import List, Tuple

import spacy
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from annotated_transformer.data.tokenizer import tokenize
from annotated_transformer.data.vocab import Vocab


class Multi30kDataset(Dataset):
    """Map-style wrapper around the HuggingFace `bentrevett/multi30k` dataset."""

    def __init__(self, split: str):
        ds = load_dataset("bentrevett/multi30k", split=split, trust_remote_code=False)
        self.data = list(ds)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return (self.data[idx]["de"], self.data[idx]["en"])


def collate_batch(
    batch: List[Tuple[str, str]],
    src_pipeline,
    tgt_pipeline,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_padding: int = 128,
    pad_id: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize, numericalize, wrap with <s>/</s>, and pad a batch of raw text pairs.

    Each raw (source_text, target_text) pair goes through the same pipeline:
    tokenize -> map tokens to vocab ids -> prepend <s> (start) / append </s>
    (end) -> right-pad to a fixed length so every sequence in the batch has
    identical shape and can be stacked into one tensor.
    """
    bs_id = torch.tensor([0], device=device)  # <s>  -- vocab index 0, see SPECIAL_TOKENS
    eos_id = torch.tensor([1], device=device)  # </s> -- vocab index 1
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        # src_pipeline/tgt_pipeline tokenize raw text (e.g. spaCy); the vocab
        # object then maps each token string to its integer id. Wrapping with
        # bs_id/eos_id lets the model learn explicit "start" and "end of
        # sequence" markers, which greedy_decode later uses to know where to
        # stop generating.
        processed_src = torch.cat(
            [bs_id, torch.tensor(src_vocab(src_pipeline(_src)), dtype=torch.int64, device=device), eos_id],
            0,
        )
        processed_tgt = torch.cat(
            [bs_id, torch.tensor(tgt_vocab(tgt_pipeline(_tgt)), dtype=torch.int64, device=device), eos_id],
            0,
        )
        # Right-pad each sequence out to max_padding with pad_id so every
        # example in the batch has the same length and can be torch.stack'd
        # into a single rectangular tensor. `(0, N)` = "pad 0 on the left,
        # N on the right" for torch.nn.functional.pad on a 1-D tensor.
        src_list.append(pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id))
        tgt_list.append(pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id))

    return torch.stack(src_list), torch.stack(tgt_list)


def create_dataloaders(
    device: torch.device,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de: spacy.Language,
    spacy_en: spacy.Language,
    batch_size: int = 12000,
    max_padding: int = 128,
    is_distributed: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation DataLoaders with tokenization + padding baked into collate_fn."""

    def tokenize_de(text: str) -> List[str]:
        return tokenize(text, spacy_de)

    def tokenize_en(text: str) -> List[str]:
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_dataset = Multi30kDataset("train")
    valid_dataset = Multi30kDataset("validation")

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader
