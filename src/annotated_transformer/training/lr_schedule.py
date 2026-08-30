"""Noam learning-rate schedule: linear warmup, then inverse-sqrt decay."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def noam_rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """LR multiplier from "Attention Is All You Need", Section 5.3.

    step is defaulted to 1 when 0 to avoid raising zero to a negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def build_noam_scheduler(
    optimizer: Optimizer, model_size: int, factor: float = 1.0, warmup: int = 3000
) -> LambdaLR:
    """Wrap an optimizer with the Noam schedule as a standard `LambdaLR`."""
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: noam_rate(step, model_size, factor, warmup),
    )
