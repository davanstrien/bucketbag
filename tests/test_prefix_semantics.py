"""Regression tests for ``prefix`` directory semantics (0.3.1).

The Hub's ``list_bucket_tree(prefix=)`` is a plain string prefix that ignores a trailing slash:
``prefix="a/b/"`` returns ``a/bc/y`` as well as ``a/b/x``. bucketbag therefore filters
client-side with the caller's *original* prefix, so a trailing ``/`` means "this directory".

Offline: ``HfApi.list_bucket_tree`` is replaced with a fake that reproduces the server's
over-matching (it matches on ``prefix.rstrip("/")``), so the tests are non-vacuous — without
the client-side filter every "directory" assertion below fails.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from conftest import bf, count_bb_dirs, make_fake_download
from huggingface_hub import HfApi

from bucketbag import batched_files, completed_keys, iter_keys

TREE = ["a/b/x", "a/bc/y", "a/c/z"]


@pytest.fixture
def fake_tree(monkeypatch):
    """Fake ``list_bucket_tree`` with the Hub's real semantics: string prefix, slash ignored.

    Returns the list of ``prefix`` values the server was called with, so tests can also check
    the server-side call stays narrow (rstripped) rather than listing the whole bucket.
    """
    calls: list[str | None] = []

    def fake(self, bucket_id, *, prefix=None, recursive=False, **kwargs):  # noqa: ANN001
        calls.append(prefix)
        head = (prefix or "").rstrip("/")
        return [bf(p, size=5) for p in TREE if p.startswith(head)]

    monkeypatch.setattr(HfApi, "list_bucket_tree", fake)
    return calls


# --- sanity: the fake really over-matches, like the Hub does -----------------------------


def test_fake_server_over_matches_trailing_slash(fake_tree):
    paths = [f.path for f in HfApi().list_bucket_tree("ns/b", prefix="a/b/", recursive=True)]
    assert paths == ["a/b/x", "a/bc/y"]


# --- iter_keys --------------------------------------------------------------------------


def test_iter_keys_trailing_slash_means_directory(fake_tree):
    assert list(iter_keys("ns/b", prefix="a/b/")) == ["a/b/x"]
    # The server call is still narrowed (rstripped), not a full-bucket scan.
    assert fake_tree == ["a/b"]


def test_iter_keys_no_slash_keeps_string_prefix_semantics(fake_tree):
    # Documented: without a trailing slash, siblings sharing the string prefix are included.
    assert list(iter_keys("ns/b", prefix="a/b")) == ["a/b/x", "a/bc/y"]


def test_iter_keys_embedded_bucket_prefix_keeps_string_semantics(fake_tree):
    # "ns/b/a/b" embeds the prefix in the bucket ref; _parse_bucket strips slashes, so this is
    # the string-prefix form. Passing prefix= explicitly is the way to get directory semantics.
    assert list(iter_keys("ns/b/a/b")) == ["a/b/x", "a/bc/y"]
    assert list(iter_keys("ns/b", prefix="a/b/", objects=True))[0].path == "a/b/x"


def test_iter_keys_include_derived_prefix_is_directory_scoped(fake_tree):
    # The literal head of the glob ("a/b/") acts as a directory prefix; the glob itself would
    # already exclude a/bc/y, so this pins that the two filters agree.
    assert list(iter_keys("ns/b", include="a/b/**")) == ["a/b/x"]
    assert fake_tree == ["a/b"]


# --- batched_files (keys derived from prefix) ---------------------------------------------


def test_batched_files_prefix_directory(fake_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download())
    batches = list(batched_files("ns/b", prefix="a/b/", prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [["a/b/x"]]
    assert count_bb_dirs(tmp_path) == 0


def test_batched_files_prefix_no_slash(fake_tree, monkeypatch, tmp_path):
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download())
    batches = list(batched_files("ns/b", prefix="a/b", prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [["a/b/x", "a/bc/y"]]


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
    shards = ["out/r1/0.parquet", "out/r10/0.parquet", "out/r1/1.parquet"]

    def fake(self, bucket_id, *, prefix=None, recursive=False, **kwargs):  # noqa: ANN001
        head = (prefix or "").rstrip("/")
        return [bf(p, size=5) for p in shards if p.startswith(head)]

    monkeypatch.setattr(HfApi, "list_bucket_tree", fake)


def test_completed_keys_prefix_directory(fake_parquet, fake_parquet_tree):
    assert completed_keys("ns/b", prefix="out/r1/") == {"out/r1/0.parquet", "out/r1/1.parquet"}


def test_completed_keys_prefix_no_slash(fake_parquet, fake_parquet_tree):
    assert completed_keys("ns/b", prefix="out/r1") == {
        "out/r1/0.parquet",
        "out/r1/1.parquet",
        "out/r10/0.parquet",
    }
