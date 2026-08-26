"""Regression tests for ``prefix`` directory semantics (0.3.1).

The Hub honors a trailing slash on the first page of a recursive listing, but its ``Link: next``
URL carries a literal ``/`` that a 302 strips, so later pages match on the slash-less string
prefix: ``prefix="a/b/"`` returns ``a/bc/y`` as well as ``a/b/x`` once the listing exceeds one
page. bucketbag sends the prefix verbatim *and* re-filters client-side with the caller's
original prefix, so a trailing ``/`` means "this directory" regardless of listing size.

Offline: ``HfApi.list_bucket_tree`` is replaced with a fake that reproduces that paginated
behaviour (``hub_like_listing``), so the tests are non-vacuous — without the client-side filter
every "directory" assertion below fails.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from conftest import bf, count_bb_dirs, make_fake_download
from huggingface_hub import HfApi

from bucketbag import batched_files, completed_keys, iter_keys

PAGE = 2  # fake page size; the real Hub pages at 1000
TREE = ["a/b/x1", "a/b/x2", "a/b/x3", "a/bc/y", "a/c/z"]
A_B = ["a/b/x1", "a/b/x2", "a/b/x3"]


def hub_like_listing(keys: list[str], prefix: str | None, page: int = PAGE) -> list[str]:
    """Reproduce ``GET /api/buckets/.../tree/<prefix>?recursive=true`` + ``Link: next`` paging.

    Page 1 is a raw string-prefix range query, trailing slash honored. The ``next`` link carries
    a literal ``/`` that the Hub 302-redirects to the slash-less path, so every later page
    matches on ``prefix.rstrip("/")`` — verified against prod on 2026-08-26 (biglam/britannica,
    ``source/pages/encyclopdiabri01chis/``: page 1 clean, page 2 crosses into ``...chisrich/``).
    """
    keys = sorted(keys)
    first = [k for k in keys if k.startswith(prefix or "")][:page]
    if len(first) < page:
        return first
    head = (prefix or "").rstrip("/")
    rest = [k for k in keys if k.startswith(head) and k > first[-1]]
    return first + rest


@pytest.fixture
def fake_tree(monkeypatch):
    """Fake ``list_bucket_tree`` with the Hub's real paginated semantics (``hub_like_listing``).

    Returns the list of ``prefix`` values the server was called with, so tests can also check
    the prefix reaches the server verbatim (slash preserved) and the call stays narrow.
    """
    calls: list[str | None] = []

    def fake(self, bucket_id, *, prefix=None, recursive=False, **kwargs):  # noqa: ANN001
        calls.append(prefix)
        return [bf(p, size=5) for p in hub_like_listing(TREE, prefix)]

    monkeypatch.setattr(HfApi, "list_bucket_tree", fake)
    return calls


# --- sanity: the fake really over-matches past page 1, like the Hub does --------------------


def test_fake_server_over_matches_after_first_page():
    assert hub_like_listing(TREE, "a/b/") == ["a/b/x1", "a/b/x2", "a/b/x3", "a/bc/y"]
    assert hub_like_listing(TREE, "a/b/", page=10) == A_B  # single page: slash honored
    assert hub_like_listing(TREE, "a/b") == ["a/b/x1", "a/b/x2", "a/b/x3", "a/bc/y"]


# --- iter_keys --------------------------------------------------------------------------


def test_iter_keys_trailing_slash_means_directory(fake_tree):
    assert list(iter_keys("ns/b", prefix="a/b/")) == A_B
    # The prefix reaches the server verbatim (slash kept) so page 1 is narrowed correctly.
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
        return [bf(p, size=5) for p in hub_like_listing(shards, prefix)]

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
