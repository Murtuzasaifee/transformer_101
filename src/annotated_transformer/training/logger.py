"""Structured training-metrics logger.

Writes one JSON object per line (JSONL) to a run-specific log file under
`log_dir`. Each line is a self-contained, timestamped metrics record, which
makes the file trivial to tail, stream, or load into a dashboard (e.g. with
`pandas.read_json(path, lines=True)`) without parsing free-text log output.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TrainingMetricsLogger:
    """Appends structured metric records to `{log_dir}/{run_name}.jsonl`."""

    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{run_name}.jsonl"
        logger.info("Training metrics will be written to %s", self.log_path)

    def log(self, event: str, **fields: Any) -> None:
        """Append one metrics record. `event` identifies the record type

        (e.g. "train_step", "eval_epoch", "checkpoint") so a dashboard can
        filter/group records without inspecting every field.
        """
        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "event": event,
            **fields,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def log_train_step(
        self,
        epoch: int,
        step: int,
        accum_step: int,
        loss: float,
        tokens_per_sec: float,
        learning_rate: float,
    ) -> None:
        self.log(
            "train_step",
            epoch=epoch,
            step=step,
            accum_step=accum_step,
            loss=loss,
            tokens_per_sec=tokens_per_sec,
            learning_rate=learning_rate,
        )

    def log_epoch_summary(
        self,
        epoch: int,
        mode: str,
        avg_loss: float,
        total_tokens: int,
        elapsed_sec: float,
    ) -> None:
        self.log(
            "epoch_summary",
            epoch=epoch,
            mode=mode,
            avg_loss=avg_loss,
            total_tokens=total_tokens,
            elapsed_sec=elapsed_sec,
        )

    def log_checkpoint(self, epoch: Optional[int], file_path: str) -> None:
        self.log("checkpoint", epoch=epoch, file_path=file_path)
