"""Typed configuration for model architecture, data, and training.

Architecture/data settings (things that change what the model *is*) live in
the YAML config. Training-run controls (things you tweak between runs without
changing the model) can be overridden via a `.env` file -- see
`ENV_OVERRIDE_VARS` below for the full list. YAML sets the baseline; `.env`
overrides it; whichever is loaded, the result is always a validated
`ExperimentConfig`.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
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
    checkpoint_every_n_epochs: int = Field(
        default=10, gt=0, description="Save a checkpoint every N epochs (a final checkpoint is always saved)"
    )
    log_dir: str = Field(default="logs")
    log_every_n_steps: int = Field(default=40, gt=0)


class ExperimentConfig(BaseModel):
    """Top-level config bundling model, data, and training settings."""

    experiment_name: str = Field(default="multi30k_transformer")
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# Maps ENV_VAR_NAME -> (TrainingConfig field name, type caster). Deliberately
# a short list: only the knobs you actually want to flip between runs without
# editing YAML (epochs, batching/accumulation, LR, checkpoint cadence,
# single- vs multi-GPU). Model architecture stays in YAML on purpose --
# changing it usually means training a genuinely different model.
ENV_OVERRIDE_VARS = {
    "NUM_EPOCHS": ("num_epochs", int),
    "BATCH_SIZE": ("batch_size", int),
    "ACCUM_ITER": ("accum_iter", int),
    "BASE_LR": ("base_lr", float),
    "WARMUP": ("warmup", int),
    "CHECKPOINT_EVERY_N_EPOCHS": ("checkpoint_every_n_epochs", int),
    "DISTRIBUTED": ("distributed", lambda v: v.strip().lower() in ("1", "true", "yes")),
}


def _apply_env_overrides(training: TrainingConfig) -> TrainingConfig:
    """Overwrite `training` fields with any matching, set env vars (from `.env` or the shell)."""
    for env_var, (field_name, cast) in ENV_OVERRIDE_VARS.items():
        raw_value = os.environ.get(env_var)
        if raw_value is not None and raw_value != "":
            setattr(training, field_name, cast(raw_value))
    return training


def load_config(path: Optional[str] = None, env_path: Optional[str] = ".env") -> ExperimentConfig:
    """Load an ExperimentConfig from YAML, then apply `.env` overrides on top.

    Precedence (lowest to highest): built-in defaults -> YAML file (`path`) ->
    `.env` file / real environment variables (`ENV_OVERRIDE_VARS`).
    """
    load_dotenv(env_path)  # no-op if the file doesn't exist

    if path is None:
        config = ExperimentConfig()
    else:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        config = ExperimentConfig(**raw)

    config.training = _apply_env_overrides(config.training)
    return config
