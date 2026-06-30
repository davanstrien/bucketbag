"""Shared fixtures and helpers for the bucketbag test suite.

The suite is **fully offline**. The network is never touched: ``batched_files`` is exercised
with explicit ``keys=`` (so no bucket listing happens) and ``HfApi.download_bucket_files`` is
replaced with an in-process fake that materializes files on local disk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from huggingface_hub import HfApi
from huggingface_hub._buckets import BucketFile


def bf(path: str, size: int = 10) -> BucketFile:
    """Build a ``BucketFile`` with just the fields bucketbag inspects (``path``, ``size``)."""
    return BucketFile(type="file", path=path, size=size, xetHash="")


def bfiles(n: int, size: int = 10, prefix: str = "dir/f") -> list[BucketFile]:
    """``n`` BucketFiles named ``<prefix>0..n-1``, each ``size`` bytes."""
    return [bf(f"{prefix}{i}.bin", size=size) for i in range(n)]


def materialize(remote: BucketFile | str, local: str) -> bytes:
    """Write deterministic bytes to ``local`` and return them. ``size`` drives length when known."""
    size = getattr(remote, "size", None) or 1
    payload = b"x" * size
    Path(local).write_bytes(payload)
    return payload


def make_fake_download(*, fail_on_call: int | None = None) -> Callable:
    """Return a fake ``HfApi.download_bucket_files``.

    Writes each file in the chunk to its local path so it "exists". If ``fail_on_call`` is set,
    the ``fail_on_call``-th invocation (1-based) raises ``RuntimeError`` *before* writing that
    chunk's files — simulating a download failure on one specific batch mid-loop.
    """
    calls = {"n": 0}

    def fake_download(self, bucket_id, *, files, token=None, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if fail_on_call is not None and calls["n"] == fail_on_call:
            raise RuntimeError(f"simulated download failure on batch #{calls['n']}")
        for remote, local in files:
            materialize(remote, local)

    return fake_download


def count_bb_dirs(base: Path) -> int:
    """How many ``bb-*`` temp dirs remain under ``base`` (the leak detector)."""
    return len(list(base.glob("bb-*")))


@pytest.fixture
def fake_download(monkeypatch):
    """Patch ``HfApi.download_bucket_files`` to materialize files locally (no network)."""
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download())
    return None
