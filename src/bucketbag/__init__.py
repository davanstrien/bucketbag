"""bucketbag — a minimal, toolz-style helper for Hugging Face bucket files.

Working with bucket-resident data means the same chore in every script: list a prefix,
download a batch of files to local disk, process them, delete them, repeat — without blowing
up disk or leaking temp files. ``huggingface_hub`` has the raw pieces; ``bucketbag`` wraps the
batch-download-process-cleanup loop into one composable verb.

The headline helper is :func:`batched_files` — think ``toolz.partition_all`` for *bucket
files*, except each batch is downloaded to a temp dir and **deleted before the next** (bounded
scratch space), with optional prefetch overlap. It is **file-type agnostic**: a
:class:`LoadedItem` is just a key + a local path + raw bytes, with opt-in lazy ``.image`` /
``.text()`` / ``.json()``.

    from bucketbag import batched_files

    for batch in batched_files("davanstrien/my-bucket", include="images/**/*.jpg", n=20):
        for it in batch:
            do_something(it.path)      # or it.bytes / it.image / it.text() / it.json()
        # <- this batch's files are auto-deleted as the loop advances

This is **not a framework**: no executors and no Jobs orchestration. The helpers compose with
plain ``for`` loops; :class:`Bag` runs the same loop as a lazy plan.

Layout: ``_core`` holds the pure rules (addressing, filters, batching, resume), ``_hub`` does
every hub call and owns the temp files, ``bag`` is the lazy plan. CONTRACT.md states the rules.
"""

from __future__ import annotations

import os

# Turn on xet's high-performance mode before huggingface_hub is imported. This is the
# documented, safe transport boost. Raising xet's per-process file-download concurrency is the
# big lever for many small files, but it over-subscribes large ones, so it stays opt-in via
# boost(). Opt out of even this default with BUCKETBAG_NO_XET_TUNE=1.
if os.environ.get("BUCKETBAG_NO_XET_TUNE", "").lower() not in ("1", "true", "yes", "on"):
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from toolz import partition_all  # noqa: E402

from ._hub import (  # noqa: E402
    LoadedItem,
    batched_files,
    boost,
    completed_keys,
    iter_keys,
    put_bytes,
    put_files,
    put_text,
)
from ._log import install as _install_log  # noqa: E402
from .bag import Bag, BagStats  # noqa: E402

_install_log()  # [bucketbag] lines on stderr until the caller configures logging

__all__ = [
    "Bag",
    "BagStats",
    "LoadedItem",
    "batched_files",
    "boost",
    "completed_keys",
    "iter_keys",
    "partition_all",
    "put_bytes",
    "put_files",
    "put_text",
]

__version__ = "0.3.1"
