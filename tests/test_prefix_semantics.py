"""Regression tests for prefix directory semantics under over-matching listings.

The Hub previously lost a trailing slash after the first page of a recursive listing.
The server bug is fixed, but the tests keep replaying it: ``FakeBucket`` with
``overmatch_after_first_page=True`` and a two-key page returns the same listing as
``hub_like_listing`` below. bucketbag must keep results within the caller's original prefix
even if an upstream listing over-matches again, so the directory assertions fail if the
client-side guards are removed.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from conftest import count_bb_dirs
from fake_bucket import FakeBucket

from bucketbag import batched_files, completed_keys, iter_keys

PAGE = 2  # fake page size; the real Hub pages at 1000
TREE = ["a/b/x1", "a/b/x2", "a/b/x3", "a/bc/y", "a/c/z"]
A_B = ["a/b/x1", "a/b/x2", "a/b/x3"]


def hub_like_listing(keys: list[str], prefix: str | None, page: int = PAGE) -> list[str]:
    """Reproduce the historical paginated over-match observed on 2026-08-26.

    Original repro: ``biglam/britannica``, prefix ``source/pages/encyclopdiabri01chis/``.

    Page 1 honors the trailing slash. Later pages simulate a next-link redirect losing
    it, so the listing spills into sibling directories. Retained as an adversarial
    regression case, independently of the current server implementation.
    """
    keys = sorted(keys)
    first = [k for k in keys if k.startswith(prefix or "")][:page]
    if len(first) < page:
        return first
    head = (prefix or "").rstrip("/")
    rest = [k for k in keys if k.startswith(head) and k > first[-1]]
    return first + rest


def overmatching_bucket(bucket: FakeBucket, keys: list[str]) -> None:
    """Store ``keys`` (5 bytes each) in ``ns/b`` and list them like ``hub_like_listing``.

    ``bucket.list_prefixes`` then records the ``prefix`` values the server was called with, so
    tests can also check the prefix reaches the server verbatim (slash preserved) and the call
    stays narrow.
    """
    bucket.page_size = PAGE
    bucket.overmatch_after_first_page = True
    bucket.seed("ns/b", keys, data=b"x" * 5)


# --- sanity: the fake reproduces the historical over-match past page 1 -------------------


def test_fake_server_over_matches_after_first_page():
    assert hub_like_listing(TREE, "a/b/") == ["a/b/x1", "a/b/x2", "a/b/x3", "a/bc/y"]
    assert hub_like_listing(TREE, "a/b/", page=10) == A_B  # single page: slash honored
    assert hub_like_listing(TREE, "a/b") == ["a/b/x1", "a/b/x2", "a/b/x3", "a/bc/y"]


# --- iter_keys --------------------------------------------------------------------------


def test_iter_keys_trailing_slash_means_directory(bucket: FakeBucket):
    overmatching_bucket(bucket, TREE)
    assert list(iter_keys("ns/b", prefix="a/b/")) == A_B
    # The prefix reaches the server verbatim (slash kept) so page 1 is narrowed correctly.
    assert bucket.list_prefixes == ["a/b/"]


def test_iter_keys_no_slash_keeps_string_prefix_semantics(bucket: FakeBucket):
    overmatching_bucket(bucket, TREE)
    # Documented: without a trailing slash, siblings sharing the string prefix are included.
    assert list(iter_keys("ns/b", prefix="a/b")) == A_B + ["a/bc/y"]


def test_iter_keys_embedded_bucket_prefix_keeps_string_semantics(bucket: FakeBucket):
    overmatching_bucket(bucket, TREE)
    # "ns/b/a/b" embeds the prefix in the bucket ref; _parse_bucket strips slashes, so this is
    # the string-prefix form. Passing prefix= explicitly is the way to get directory semantics.
    assert list(iter_keys("ns/b/a/b")) == A_B + ["a/bc/y"]
    assert [f.path for f in iter_keys("ns/b", prefix="a/b/", objects=True)] == A_B


def test_iter_keys_include_derived_prefix_is_directory_scoped(bucket: FakeBucket):
    overmatching_bucket(bucket, TREE)
    # The literal head of the glob ("a/b/") is sent as the prefix; the glob itself would
    # already exclude a/bc/y, so this pins that the two filters agree.
    assert list(iter_keys("ns/b", include="a/b/**")) == A_B
    assert bucket.list_prefixes == ["a/b/"]


# --- batched_files (keys derived from prefix) ---------------------------------------------


def test_batched_files_prefix_directory(bucket: FakeBucket, tmp_path):
    overmatching_bucket(bucket, TREE)
    batches = list(batched_files("ns/b", prefix="a/b/", prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [A_B]
    assert count_bb_dirs(tmp_path) == 0


def test_batched_files_prefix_no_slash(bucket: FakeBucket, tmp_path):
    overmatching_bucket(bucket, TREE)
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


SHARDS = ["out/r1/0.parquet", "out/r1/1.parquet", "out/r1/2.parquet", "out/r10/0.parquet"]


def test_completed_keys_prefix_directory(fake_parquet, bucket: FakeBucket):
    overmatching_bucket(bucket, SHARDS)
    assert completed_keys("ns/b", prefix="out/r1/") == {
        "out/r1/0.parquet",
        "out/r1/1.parquet",
        "out/r1/2.parquet",
    }


def test_completed_keys_prefix_no_slash(fake_parquet, bucket: FakeBucket):
    overmatching_bucket(bucket, SHARDS)
    assert completed_keys("ns/b", prefix="out/r1") == {
        "out/r1/0.parquet",
        "out/r1/1.parquet",
        "out/r1/2.parquet",
        "out/r10/0.parquet",
    }
