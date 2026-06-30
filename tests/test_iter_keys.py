"""Tests for ``iter_keys`` — the listing + glob-filter + sort path.

Offline: ``HfApi.list_bucket_tree`` is monkeypatched to return in-memory entries, so no bucket is
ever listed over the network. This exercises the ``found.sort(...)`` guarantee, the include-glob
filter, and the non-file skip in ``_list_bucketfiles`` — none of which the ``batched_files(keys=)``
tests can reach, because passing ``keys=`` hands the keys straight through, unsorted and unfiltered.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import bf
from huggingface_hub import HfApi

from bucketbag import iter_keys


@pytest.fixture
def fake_list(monkeypatch):
    """Install a fake ``HfApi.list_bucket_tree`` that yields a fixed set of entries (no network)."""

    def _install(entries):
        def fake(self, bucket_id, *, prefix=None, recursive=False, **kwargs):
            return list(entries)

        monkeypatch.setattr(HfApi, "list_bucket_tree", fake)

    return _install


def test_iter_keys_returns_sorted(fake_list):
    # Deliberately out of lexical order on input; iter_keys must return them sorted.
    fake_list([bf("b/2.txt"), bf("a/1.txt"), bf("a/10.txt"), bf("a/2.txt")])
    assert list(iter_keys("ns/bucket")) == ["a/1.txt", "a/10.txt", "a/2.txt", "b/2.txt"]


def test_iter_keys_applies_include_glob(fake_list):
    fake_list([bf("a/1.jp2"), bf("a/2.txt"), bf("a/3.jp2")])
    assert list(iter_keys("ns/bucket", include="**/*.jp2")) == ["a/1.jp2", "a/3.jp2"]


def test_iter_keys_skips_non_files(fake_list):
    # A directory-like entry (type != "file") must be dropped before sorting/yielding.
    fake_list([bf("a/1.txt"), SimpleNamespace(type="directory", path="a")])
    assert list(iter_keys("ns/bucket")) == ["a/1.txt"]
