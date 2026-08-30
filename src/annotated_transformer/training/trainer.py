"""Training loop: single-epoch driver plus the multi-GPU (DDP) training entrypoint."""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from annotated_transformer.config import ExperimentConfig
from annotated_transformer.data.batch import Batch
from annotated_transformer.data.dataset import create_dataloaders
from annotated_transformer.data.vocab import Vocab
from annotated_transformer.model.transformer import build_transformer_model
from annotated_transformer.training.logger import TrainingMetricsLogger
from annotated_transformer.training.loss import LabelSmoothing, SimpleLossCompute
from annotated_transformer.training.lr_schedule import build_noam_scheduler

logger = logging.getLogger(__name__)


class DummyOptimizer(torch.optim.Optimizer):
    """No-op optimizer used during evaluation, so eval can reuse the train loop code path."""

    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none: bool = False):
        pass


class DummyScheduler:
    """No-op LR scheduler, paired with DummyOptimizer for evaluation passes."""

    def step(self):
        pass


@dataclass
class TrainState:
    """Running counters tracked across an entire training run."""

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter: Iterable[Batch],
    model: nn.Module,
    loss_compute: SimpleLossCompute,
    optimizer,
    scheduler,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState = None,
    metrics_logger: Optional[TrainingMetricsLogger] = None,
    epoch: int = 0,
    log_every_n_steps: int = 40,
):
    """Run one pass over `data_iter`, optionally updating weights (mode == "train*")."""
    if train_state is None:
        train_state = TrainState()

    epoch_start = time.time()
    start = time.time()
    total_tokens = 0
    total_loss = 0.0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        # Full forward pass: encode src, decode tgt, project to vocab logits.
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode in ("train", "train+log"):
            # Backprop accumulates gradients into .grad on every parameter
            # (it does NOT update weights by itself -- optimizer.step() does).
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            # Gradient accumulation: instead of updating weights after every
            # single batch, we let gradients build up over `accum_iter`
            # batches before applying them. This approximates training with a
            # much larger batch size than would otherwise fit in GPU memory,
            # since only one batch's activations need to be held at a time.
            if i % accum_iter == 0:
                optimizer.step()  # apply the accumulated gradients to the weights
                optimizer.zero_grad(set_to_none=True)  # clear .grad before the next accumulation window
                n_accum += 1
                train_state.accum_step += 1

            # The LR scheduler still steps every batch (not just every
            # accumulation window) so the warmup/decay curve is driven by
            # wall-clock batches processed, matching the reference schedule.
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % log_every_n_steps == 1 and mode in ("train", "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tok_per_sec = float(tokens) / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f | "
                "Tokens/Sec: %7.1f | Learning Rate: %6.1e",
                i,
                n_accum,
                loss / batch.ntokens,
                tok_per_sec,
                lr,
            )
            if metrics_logger is not None:
                metrics_logger.log_train_step(
                    epoch=epoch,
                    step=train_state.step,
                    accum_step=train_state.accum_step,
                    loss=float(loss / batch.ntokens),
                    tokens_per_sec=tok_per_sec,
                    learning_rate=lr,
                )
            start = time.time()
            tokens = 0

        del loss, loss_node

    avg_loss = total_loss / total_tokens
    if metrics_logger is not None:
        metrics_logger.log_epoch_summary(
            epoch=epoch,
            mode=mode,
            avg_loss=float(avg_loss),
            total_tokens=int(total_tokens),
            elapsed_sec=time.time() - epoch_start,
        )
    return avg_loss, train_state


def train_worker(
    gpu: int,
    ngpus_per_node: int,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    spacy_de,
    spacy_en,
    config: ExperimentConfig,
    is_distributed: bool = False,
) -> None:
    """Train (or DDP-train, one process per GPU) the model for `config.training.num_epochs`."""
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")
    logger.info("Train worker process using device: %s", device)

    if cuda_available:
        torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = config.model.d_model
    model = build_transformer_model(
        len(vocab_src),
        len(vocab_tgt),
        n_layers=config.model.n_layers,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
    )
    model = model.to(device)
    # `module` always points at the *underlying* model with real attributes
    # like `.generator` -- when wrapped in DDP below, `model` itself becomes a
    # DDP wrapper whose attributes proxy through `.module`, so we keep this
    # separate handle to avoid repeating `.module` everywhere.
    module = model
    is_main_process = True

    if is_distributed:
        # One process per GPU, all synchronized via NCCL (NVIDIA's collective
        # communications library). Each process only sees its own `gpu` id.
        dist.init_process_group("nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node)
        # DDP wraps the model so gradients are automatically averaged across
        # all GPUs after every backward() call -- each GPU trains on a
        # different shard of the data but all end up with identical weights.
        model = DDP(model, device_ids=[gpu])
        module = model.module
        # Only GPU 0 does "shared" work (checkpointing, metrics logging) to
        # avoid every process writing the same file simultaneously.
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=config.training.label_smoothing
    ).to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config.training.batch_size // ngpus_per_node,
        max_padding=config.data.max_padding,
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = build_noam_scheduler(
        optimizer, model_size=d_model, factor=1.0, warmup=config.training.warmup
    )
    train_state = TrainState()

    metrics_logger = (
        TrainingMetricsLogger(config.training.log_dir, f"{config.experiment_name}_gpu{gpu}")
        if is_main_process
        else None
    )

    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    for epoch in range(config.training.num_epochs):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        logger.info("[GPU%d] Epoch %d Training ====", gpu, epoch)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config.training.accum_iter,
            train_state=train_state,
            metrics_logger=metrics_logger,
            epoch=epoch,
            log_every_n_steps=config.training.log_every_n_steps,
        )

        if cuda_available:
            torch.cuda.empty_cache()

        if is_main_process:
            file_path = os.path.join(
                config.training.checkpoint_dir,
                f"{config.training.checkpoint_prefix}{epoch:02d}.pt",
            )
            torch.save(module.state_dict(), file_path)
            if metrics_logger is not None:
                metrics_logger.log_checkpoint(epoch=epoch, file_path=file_path)

        logger.info("[GPU%d] Epoch %d Validation ====", gpu, epoch)
        model.eval()
        eval_loss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
            metrics_logger=metrics_logger,
            epoch=epoch,
        )
        logger.info("[GPU%d] Epoch %d Validation Loss: %.4f", gpu, epoch, eval_loss)
        if cuda_available:
            torch.cuda.empty_cache()

    if is_main_process:
        file_path = os.path.join(
            config.training.checkpoint_dir, f"{config.training.checkpoint_prefix}final.pt"
        )
        torch.save(module.state_dict(), file_path)
        if metrics_logger is not None:
            metrics_logger.log_checkpoint(epoch=None, file_path=file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config: ExperimentConfig) -> None:
    """Spawn one training process per visible GPU and run them under DistributedDataParallel."""
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    logger.info("Number of GPUs detected: %d", ngpus)
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config: ExperimentConfig) -> None:
    """Entrypoint: dispatch to single-GPU/CPU or multi-GPU (DDP) training based on config."""
    if config.training.distributed:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, is_distributed=False)
