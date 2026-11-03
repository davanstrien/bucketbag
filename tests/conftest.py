"""Shared fixtures and helpers for the bucketbag test suite.

The suite is **fully offline**. The ``bucket`` fixture replaces the three ``HfApi`` bucket
methods with one in-memory store (``fake_bucket.FakeBucket``), so listings, downloads and
uploads all see the same state.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
from fake_bucket import FakeBucket
from huggingface_hub import BucketFile


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "internals: reaches private names (_pack, BucketRef, _glob_*, LoadedItem's byte cache). "
        "A restructure may rewrite these tests; `pytest -m 'not internals'` must pass unchanged.",
    )


class KnownGap(AssertionError):
    """A contract check that today's code fails. Raised only through :func:`gap`."""


def gap(message: str) -> NoReturn:
    """Report that the code breaks the contract under test, in a way already recorded."""
    raise KnownGap(message)


def known_gap(reason: str) -> pytest.MarkDecorator:
    """Mark a test that checks a contract today's code breaks.

    Only ``KnownGap`` counts as the expected failure: an assertion error, a harness error or a
    timeout still fails the test. ``strict=True`` turns a pass into a failure, so the marker
    has to come off in the PR that fixes the gap.
    """
    return pytest.mark.xfail(raises=KnownGap, strict=True, reason=reason)


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch) -> FakeBucket:
    """An empty in-memory Hub (see ``fake_bucket.py``), installed for this test."""
    return FakeBucket().install(monkeypatch)


def bf(path: str, size: int = 10) -> BucketFile:
    """Build a ``BucketFile`` with just the fields bucketbag inspects (``path``, ``size``)."""
    return BucketFile(type="file", path=path, size=size, xetHash="")


def count_bb_dirs(base: Path) -> int:
    """How many ``bb-*`` temp dirs remain under ``base`` (the leak detector)."""
    return len(list(base.glob("bb-*")))


@pytest.fixture
def fake_parquet(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stand in for pyarrow + HfFileSystem, so ``completed_keys`` runs offline without pyarrow.

    Each ``<key>.parquet`` shard "contains" one ``__source_key`` equal to its own key, so the
    returned set tells us which shards were read. Set ``columns`` to change every shard's
    schema; add a key to ``unreadable`` to make that shard fail to read as parquet.
    """
    state = SimpleNamespace(columns=["__source_key"], unreadable=set())

    class Handle:
        def __init__(self, key: str) -> None:
            self.key = key

    class FileSystem:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        @contextmanager
        def open(self, url: str, mode: str = "rb") -> Generator[Handle, None, None]:
            yield Handle(url.removeprefix("hf://buckets/").split("/", 2)[2])

    class ParquetFile:
        def __init__(self, fh: Handle) -> None:
            if fh.key in state.unreadable:  # a corrupt shard fails to open, as in pyarrow
                raise ValueError(f"{fh.key} is not a parquet file")
            self.key = fh.key
            self.schema_arrow = SimpleNamespace(names=list(state.columns))

        def read(self, columns: list[str]) -> SimpleNamespace:
            missing = [name for name in columns if name not in state.columns]
            if missing:  # pyarrow raises for a column the file does not have
                raise ValueError(f"No match for FieldRef.Name({missing[0]})")
            key = self.key
            return SimpleNamespace(column=lambda name: SimpleNamespace(to_pylist=lambda: [key]))

    pq = SimpleNamespace(ParquetFile=ParquetFile)
    # ``import pyarrow.parquet as pq`` binds via the parent's attribute, so both must agree.
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=pq))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FileSystem)
    return state
