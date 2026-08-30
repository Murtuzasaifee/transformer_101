"""Console logging configuration shared by all CLI scripts."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a single readable stream handler.

    Safe to call multiple times (e.g. once per script, once per DDP worker) --
    it clears any existing handlers first to avoid duplicate log lines.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
