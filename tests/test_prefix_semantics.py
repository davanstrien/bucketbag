"""Regression tests for directory and string-prefix semantics.

The Hub preserves a trailing slash across recursive listing pages. These offline tests fake
that API contract and check that bucketbag forwards prefixes verbatim for listing, batching
and resume scans.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from conftest import bf, count_bb_dirs, make_fake_download
from huggingface_hub import HfApi

from bucketbag import batched_files, completed_keys, iter_keys

TREE = ["a/b/x1", "a/b/x2", "a/b/x3", "a/bc/y", "a/c/z"]
A_B = ["a/b/x1", "a/b/x2", "a/b/x3"]


@pytest.fixture
def fake_tree(monkeypatch):
    """Fake prefix-scoped listings and record the exact prefixes sent to the Hub."""
    calls: list[str | None] = []

    def fake(self, bucket_id, *, prefix=None, recursive=False, **kwargs):  # noqa: ANN001
        calls.append(prefix)
        return [bf(p, size=5) for p in sorted(TREE) if p.startswith(prefix or "")]

    monkeypatch.setattr(HfApi, "list_bucket_tree", fake)
    return calls


# --- iter_keys --------------------------------------------------------------------------


def test_iter_keys_trailing_slash_means_directory(fake_tree):
    assert list(iter_keys("ns/b", prefix="a/b/")) == A_B
    # The prefix reaches the server verbatim, including the trailing slash.
    assert fake_tree == ["a/b/"]


def test_iter_keys_no_slash_keeps_string_prefix_semantics(fake_tree):
    # Documented: without a trailing slash, siblings sharing the string prefix are included.
    assert list(iter_keys("ns/b", prefix="a/b")) == A_B + ["a/bc/y"]


def test_iter_keys_embedded_bucket_prefix_keeps_string_semantics(fake_tree):
    # "ns/b/a/b" embeds the prefix in the bucket ref; _parse_bucket strips slashes, so this is
    # the string-prefix form. Passing prefix= explicitly is the way to get directory semantics.
    assert list(iter_keys("ns/b/a/b")) == A_B + ["a/bc/y"]
    assert [f.path for f in iter_keys("ns/b", prefix="a/b/", objects=True)] == A_B


def test_iter_keys_include_derived_prefix_is_directory_scoped(fake_tree):
    # The literal head of the glob ("a/b/") is sent as the prefix; the glob itself would
    # already exclude a/bc/y, so this pins that the two filters agree.
    assert list(iter_keys("ns/b", include="a/b/**")) == A_B
    assert fake_tree == ["a/b/"]


# --- batched_files (keys derived from prefix) ---------------------------------------------


def test_batched_files_prefix_directory(fake_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download())
    batches = list(batched_files("ns/b", prefix="a/b/", prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [A_B]
    assert count_bb_dirs(tmp_path) == 0


def test_batched_files_prefix_no_slash(fake_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download())
    batches = list(batched_files("ns/b", prefix="a/b", prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [A_B + ["a/bc/y"]]


# --- completed_keys(prefix=) --------------------------------------------------------------


@pytest.fixture
def fake_parquet(monkeypatch):
    """Stand in for pyarrow + HfFileSystem so ``completed_keys`` runs offline without pyarrow.

    Each ``<key>.parquet`` shard "contains" one ``__source_key`` equal to its own path, so the
    returned set tells us exactly which shards were scanned.
    """

    class _Handle:
        def __init__(self, path: str) -> None:
            self.path = path

    class _FS:
        def __init__(self, *a, **k) -> None:
            pass

        @contextmanager
        def open(self, full: str, mode: str = "rb"):
            yield _Handle(full.removeprefix("hf://buckets/ns/b/"))

    class _Table:
        def __init__(self, vals: list[str]) -> None:
            self._vals = vals

        def column(self, name: str):
            return SimpleNamespace(to_pylist=lambda: self._vals)

    def read_table(fh: _Handle, columns=None):  # noqa: ANN001
        return _Table([fh.path])

    pq = SimpleNamespace(read_table=read_table)
    # ``import pyarrow.parquet as pq`` binds via the parent's attribute, so both must agree.
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=pq))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", _FS)


@pytest.fixture
def fake_parquet_tree(monkeypatch):
    shards = ["out/r1/0.parquet", "out/r1/1.parquet", "out/r1/2.parquet", "out/r10/0.parquet"]

    def fake(self, bucket_id, *, prefix=None, recursive=False, **kwargs):  # noqa: ANN001
        return [bf(p, size=5) for p in sorted(shards) if p.startswith(prefix or "")]

    monkeypatch.setattr(HfApi, "list_bucket_tree", fake)


def test_completed_keys_prefix_directory(fake_parquet, fake_parquet_tree):
    assert completed_keys("ns/b", prefix="out/r1/") == {
        "out/r1/0.parquet",
        "out/r1/1.parquet",
        "out/r1/2.parquet",
    }


def test_completed_keys_prefix_no_slash(fake_parquet, fake_parquet_tree):
    assert completed_keys("ns/b", prefix="out/r1") == {
        "out/r1/0.parquet",
        "out/r1/1.parquet",
        "out/r1/2.parquet",
        "out/r10/0.parquet",
    }
