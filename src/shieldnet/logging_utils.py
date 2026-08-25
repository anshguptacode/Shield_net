"""Logging and progress helpers.

One place to configure formatting so every stage of a long training run is traceable
after the fact, and a :func:`stage` context manager that times each phase.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

__all__ = ["get_logger", "configure_logging", "stage", "human_duration", "human_count"]

_CONFIGURED = False
_FMT = "%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    logfile: Optional[os.PathLike | str] = None,
    force: bool = False,
) -> None:
    """Install a stream handler (and optionally a file handler) exactly once.

    Also quiets the third-party loggers that otherwise bury our own output: Optuna
    logs a line per trial, TensorFlow logs device placement, and matplotlib logs every
    font it considers.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger("shieldnet")
    root.handlers.clear()
    root.setLevel(level)
    root.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(stream)

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(file_handler)

    for noisy, lvl in (
        ("matplotlib", logging.WARNING),
        ("matplotlib.font_manager", logging.ERROR),
        ("optuna", logging.WARNING),
        ("shap", logging.WARNING),
        ("numexpr", logging.WARNING),
        ("PIL", logging.WARNING),
    ):
        logging.getLogger(noisy).setLevel(lvl)

    # TensorFlow's C++ layer reads this before the Python logger exists, so it has to
    # be an env var and it has to be set before `import tensorflow`.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``shieldnet`` namespace."""
    configure_logging()
    short = name.split(".")[-1] if name.startswith("shieldnet") else name
    return logging.getLogger(f"shieldnet.{short}")


def human_duration(seconds: float) -> str:
    """``93.4`` -> ``'1m 33s'``; ``0.7`` -> ``'0.7s'``."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def human_count(n: float) -> str:
    """``304616`` -> ``'304,616'``, tolerating floats and None-ish values."""
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


@contextmanager
def stage(logger: logging.Logger, title: str, *, quiet: bool = False) -> Iterator[dict]:
    """Time a pipeline phase and log its start and end.

    Yields a dict the body can stash results in; anything placed under the key
    ``summary`` is appended to the completion line.

        with stage(log, "Building working chunk") as st:
            df = build_chunk(...)
            st["summary"] = f"{len(df):,} rows"
    """
    box: dict = {}
    if not quiet:
        logger.info("=" * 74)
        logger.info("%s ...", title)
    started = time.perf_counter()
    try:
        yield box
    except Exception:
        logger.error("%s FAILED after %s", title, human_duration(time.perf_counter() - started))
        raise
    elapsed = time.perf_counter() - started
    summary = box.get("summary")
    if not quiet:
        if summary:
            logger.info("%s done in %s - %s", title, human_duration(elapsed), summary)
        else:
            logger.info("%s done in %s", title, human_duration(elapsed))
    box["elapsed"] = elapsed
