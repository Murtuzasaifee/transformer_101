"""Training loop, loss, learning-rate schedule, and structured metrics logging."""

from annotated_transformer.training.logger import TrainingMetricsLogger
from annotated_transformer.training.loss import LabelSmoothing, SimpleLossCompute
from annotated_transformer.training.lr_schedule import build_noam_scheduler, noam_rate
from annotated_transformer.training.trainer import (
    DummyOptimizer,
    DummyScheduler,
    TrainState,
    run_epoch,
    train_distributed_model,
    train_model,
    train_worker,
)

__all__ = [
    "TrainingMetricsLogger",
    "LabelSmoothing",
    "SimpleLossCompute",
    "build_noam_scheduler",
    "noam_rate",
    "DummyOptimizer",
    "DummyScheduler",
    "TrainState",
    "run_epoch",
    "train_distributed_model",
    "train_model",
    "train_worker",
]
