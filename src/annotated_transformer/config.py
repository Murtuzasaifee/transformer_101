"""Typed configuration for model architecture, data, and training -- loaded from YAML."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Transformer architecture hyperparameters."""

    n_layers: int = Field(default=6, description="Number of encoder/decoder layers (N)")
    d_model: int = Field(default=512, description="Embedding / hidden dimension")
    d_ff: int = Field(default=2048, description="Feed-forward inner dimension")
    n_heads: int = Field(default=8, description="Number of attention heads")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class DataConfig(BaseModel):
    """Dataset and tokenization settings."""

    dataset_name: str = Field(default="bentrevett/multi30k")
    source_lang: str = Field(default="de")
    target_lang: str = Field(default="en")
    max_padding: int = Field(default=128, description="Fixed sequence length after padding")
    vocab_cache_path: str = Field(default="checkpoints/vocab.pt")


class TrainingConfig(BaseModel):
    """Optimization loop settings."""

    num_epochs: int = Field(default=8, gt=0)
    batch_size: int = Field(default=32, gt=0)
    accum_iter: int = Field(default=10, gt=0, description="Gradient accumulation steps")
    base_lr: float = Field(default=1.0, gt=0.0, description="Peak LR scale factor (Noam schedule)")
    warmup: int = Field(default=3000, gt=0, description="Warmup steps for the LR schedule")
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    distributed: bool = Field(default=False, description="Use DistributedDataParallel across all visible GPUs")
    checkpoint_dir: str = Field(default="checkpoints")
    checkpoint_prefix: str = Field(default="multi30k_model_")
    log_dir: str = Field(default="logs")
    log_every_n_steps: int = Field(default=40, gt=0)


class ExperimentConfig(BaseModel):
    """Top-level config bundling model, data, and training settings."""

    experiment_name: str = Field(default="multi30k_transformer")
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file, falling back to defaults if no path is given."""
    if path is None:
        return ExperimentConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return ExperimentConfig(**raw)
