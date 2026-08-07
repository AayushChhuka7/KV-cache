"""Small logging helpers that produce consistent, paper-style output."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterator, Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_initialized = False


def _configure_root() -> None:
    global _initialized
    if _initialized:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(os.environ.get("KVBENCH_LOG_LEVEL", "INFO").upper())
    _initialized = True


def get_logger(name: str = "kvbench") -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


@contextmanager
def log_section(title: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Context manager that prints a banner around a block of work.

    Example
    -------
    >>> with log_section("Benchmark: Baseline FP16"):
    ...     ...
    """
    if logger is None:
        logger = get_logger()
    bar = "=" * max(8, min(60, len(title) + 4))
    logger.info(bar)
    logger.info("  " + title)
    logger.info(bar)
    try:
        yield
    finally:
        logger.info(bar)
        logger.info("")
