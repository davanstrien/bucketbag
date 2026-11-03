"""Shared fixtures and helpers for the bucketbag test suite.

The suite is **fully offline**. The ``bucket`` fixture replaces the three ``HfApi`` bucket
methods with one in-memory store (``fake_bucket.FakeBucket``), so listings, downloads and
uploads all see the same state.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
from fake_bucket import FakeBucket
from huggingface_hub import BucketFile


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "internals: reaches private names (_pack, _parse_bucket, _glob_*, LoadedItem._bytes). "
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
