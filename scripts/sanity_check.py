#!/usr/bin/env python
"""Sanity check: verify the annotated_transformer package is wired together correctly.

Runs fast, CPU-only checks that don't need the Multi30k dataset or a cached
vocabulary -- meant to be run right after `uv sync` (locally or on a fresh
GPU server) to catch import errors, shape mismatches, or config regressions
before kicking off a real training job.

Checks:
    1. Every submodule imports cleanly.
    2. A small model builds and does a forward pass with the right output shape.
    3. The training loop (run_epoch) runs on synthetic copy-task data, in both
       train and eval mode, without crashing.
    4. Greedy decoding produces a sequence of the requested length.
    5. The structured JSONL metrics logger writes valid, parseable records.
    6. The bundled YAML config loads and validates.

Usage:
    uv run scripts/sanity_check.py
"""

import json
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger("sanity_check")


class CheckFailed(Exception):
    pass


def check_imports() -> None:
    import annotated_transformer  # noqa: F401
    from annotated_transformer.model import build_transformer_model  # noqa: F401
    from annotated_transformer.data import Batch, Vocab  # noqa: F401
    from annotated_transformer.training import (  # noqa: F401
        LabelSmoothing,
        SimpleLossCompute,
        TrainState,
        build_noam_scheduler,
        run_epoch,
    )
    from annotated_transformer.inference import greedy_decode  # noqa: F401
    from annotated_transformer.config import load_config  # noqa: F401


def check_model_forward_pass() -> None:
    import torch

    from annotated_transformer.model import build_transformer_model
    from annotated_transformer.model.mask import subsequent_mask

    vocab_size = 11
    model = build_transformer_model(vocab_size, vocab_size, n_layers=2, d_model=32, d_ff=64, n_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params == 0:
        raise CheckFailed("model has zero parameters")

    src = torch.randint(1, vocab_size, (2, 10))
    tgt = torch.randint(1, vocab_size, (2, 9))
    src_mask = torch.ones(2, 1, 10)
    tgt_mask = subsequent_mask(9).expand(2, -1, -1)

    out = model(src, tgt, src_mask, tgt_mask)
    expected_shape = (2, 9, 32)
    if tuple(out.shape) != expected_shape:
        raise CheckFailed(f"forward pass output shape {tuple(out.shape)} != {expected_shape}")

    logits = model.generator(out)
    if logits.shape[-1] != vocab_size:
        raise CheckFailed(f"generator output vocab dim {logits.shape[-1]} != {vocab_size}")


def check_training_loop(tmp_log_dir: str) -> None:
    import torch

    from annotated_transformer.data.batch import Batch
    from annotated_transformer.model import build_transformer_model
    from annotated_transformer.training import (
        DummyOptimizer,
        DummyScheduler,
        LabelSmoothing,
        SimpleLossCompute,
        TrainState,
        build_noam_scheduler,
        run_epoch,
    )
    from annotated_transformer.training.logger import TrainingMetricsLogger

    vocab_size = 11

    def synthetic_batches(n_batches: int, batch_size: int = 8):
        for _ in range(n_batches):
            data = torch.randint(1, vocab_size, size=(batch_size, 10))
            data[:, 0] = 1
            yield Batch(data.clone(), data.clone(), pad=0)

    model = build_transformer_model(vocab_size, vocab_size, n_layers=2, d_model=32, d_ff=64, n_heads=4)
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = build_noam_scheduler(optimizer, model_size=32, factor=1.0, warmup=50)
    metrics_logger = TrainingMetricsLogger(tmp_log_dir, "sanity_check")

    model.train()
    train_loss, train_state = run_epoch(
        synthetic_batches(3),
        model,
        SimpleLossCompute(model.generator, criterion),
        optimizer,
        lr_scheduler,
        mode="train+log",
        train_state=TrainState(),
        metrics_logger=metrics_logger,
        epoch=0,
        log_every_n_steps=1,
    )
    if not torch.isfinite(train_loss):
        raise CheckFailed(f"train loss is not finite: {train_loss}")
    if train_state.step != 3:
        raise CheckFailed(f"expected 3 train steps, got {train_state.step}")

    model.eval()
    eval_loss, _ = run_epoch(
        synthetic_batches(2),
        model,
        SimpleLossCompute(model.generator, criterion),
        DummyOptimizer(),
        DummyScheduler(),
        mode="eval",
        metrics_logger=metrics_logger,
        epoch=0,
    )
    if not torch.isfinite(eval_loss):
        raise CheckFailed(f"eval loss is not finite: {eval_loss}")

    log_path = Path(tmp_log_dir) / "sanity_check.jsonl"
    if not log_path.exists():
        raise CheckFailed(f"metrics log file was not created at {log_path}")
    lines = log_path.read_text().strip().splitlines()
    if not lines:
        raise CheckFailed("metrics log file is empty")
    for line in lines:
        record = json.loads(line)  # raises if not valid JSON
        if "event" not in record or "timestamp" not in record:
            raise CheckFailed(f"metrics record missing required fields: {record}")


def check_greedy_decode() -> None:
    import torch

    from annotated_transformer.inference import greedy_decode
    from annotated_transformer.model import build_transformer_model

    vocab_size = 11
    model = build_transformer_model(vocab_size, vocab_size, n_layers=2, d_model=32, d_ff=64, n_heads=4)
    model.eval()

    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    src_mask = torch.ones(1, 1, 10)
    out = greedy_decode(model, src, src_mask, max_len=10, start_symbol=0)
    if out.shape != (1, 10):
        raise CheckFailed(f"greedy_decode output shape {tuple(out.shape)} != (1, 10)")


def check_config_loads() -> None:
    from annotated_transformer.config import load_config

    repo_root = Path(__file__).resolve().parents[1]
    default_config_path = repo_root / "configs" / "multi30k.yaml"

    config = load_config(str(default_config_path)) if default_config_path.exists() else load_config()
    if config.model.d_model <= 0 or config.model.n_layers <= 0:
        raise CheckFailed(f"config has invalid model hyperparameters: {config.model}")
    if config.training.num_epochs <= 0:
        raise CheckFailed(f"config has invalid training hyperparameters: {config.training}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checks = [
        ("Package imports", check_imports),
        ("Model forward pass", check_model_forward_pass),
        ("Config loads", check_config_loads),
        ("Greedy decoding", check_greedy_decode),
    ]

    tmp_log_dir = tempfile.mkdtemp(prefix="annotated_transformer_sanity_")
    checks.insert(3, ("Training loop + metrics logging", lambda: check_training_loop(tmp_log_dir)))

    failures = []
    for name, fn in checks:
        try:
            fn()
            logger.info("[PASS] %s", name)
        except Exception as exc:  # noqa: BLE001 -- want to catch and report everything
            logger.error("[FAIL] %s: %s", name, exc)
            logger.debug(traceback.format_exc())
            failures.append(name)

    shutil.rmtree(tmp_log_dir, ignore_errors=True)

    logger.info("-" * 60)
    if failures:
        logger.error("Sanity check FAILED (%d/%d checks failed): %s", len(failures), len(checks), failures)
        sys.exit(1)
    else:
        logger.info("Sanity check PASSED (%d/%d checks). Safe to start training.", len(checks), len(checks))


if __name__ == "__main__":
    main()
