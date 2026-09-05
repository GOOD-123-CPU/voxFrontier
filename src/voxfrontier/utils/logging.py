"""Logging helpers shared by the CLI and pipeline stages."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "voxfrontier", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent handler setup)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_verbose(verbose: bool) -> None:
    logging.getLogger("voxfrontier").setLevel(logging.DEBUG if verbose else logging.INFO)
