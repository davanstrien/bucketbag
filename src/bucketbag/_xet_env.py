"""xet transport settings that bucketbag sets for the whole process.

xet reads these environment variables once, when its runtime starts (the first download), and
huggingface_hub reads ``HF_XET_HIGH_PERFORMANCE`` when it is imported. So bucketbag sets them
when it is imported, before either, and only when the caller has not set them. They apply to
every Hugging Face download in the process, not only bucketbag's, so :func:`log_once` reports
them the first time bucketbag downloads.

Why these values: xet allows 8 concurrent file downloads per process by default, one shared
limit for every download call. On 0.8-2.5 MB files a limit of 32 doubled throughput (HF Jobs
benchmark, 2026-11-05); on 0.5-0.7 GB files it changed nothing and raised peak memory by about
1 GB. xet caps its download buffer memory either way (half of usable memory in
high-performance mode).

This module imports nothing from huggingface_hub, so ``__init__`` can run it first.
"""

from __future__ import annotations

import logging
import os

OPT_OUT = "BUCKETBAG_NO_XET_TUNE"
HIGH_PERFORMANCE = "HF_XET_HIGH_PERFORMANCE"
FILE_CONCURRENCY = "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"
DEFAULTS = {HIGH_PERFORMANCE: "1", FILE_CONCURRENCY: "32"}
XET_FILE_CONCURRENCY_DEFAULT = 8  # xet's own default, for the log line

_set_by_bucketbag: dict[str, str] = {}
_logged = False


def opted_out() -> bool:
    """Whether the caller asked bucketbag to leave xet's settings alone."""
    return os.environ.get(OPT_OUT, "").lower() in ("1", "true", "yes", "on")


def apply_defaults() -> None:
    """Set each default the caller has not set. Called once, at import."""
    if opted_out():
        return
    for name, value in DEFAULTS.items():
        if name not in os.environ:
            os.environ[name] = value
            _set_by_bucketbag[name] = value


def set_value(name: str, value: str) -> None:
    """Set ``name`` unless the caller exported it. bucketbag's own default may be replaced."""
    if name in os.environ and name not in _set_by_bucketbag:
        return
    os.environ[name] = value
    _set_by_bucketbag[name] = value


def describe() -> str:
    """One line: each setting's value and who set it."""
    parts = []
    for name in DEFAULTS:
        value = os.environ.get(name)
        if name in _set_by_bucketbag and value == _set_by_bucketbag[name]:
            source = "set by bucketbag"
        elif value is None:
            source = "not set, xet default"
        else:
            source = "from your environment"
        parts.append(f"{name}={value if value is not None else '-'} ({source})")
    return (
        "xet settings for this process: "
        + ", ".join(parts)
        + ". They apply to every Hugging Face download in this process. Export your own values "
        f"before importing bucketbag, or set {OPT_OUT}=1 to keep xet's defaults "
        f"({XET_FILE_CONCURRENCY_DEFAULT} concurrent files)."
    )


def log_once(logger: logging.Logger) -> None:
    """Log :func:`describe` at INFO, the first time only."""
    global _logged
    if _logged:
        return
    _logged = True
    logger.info(describe())
