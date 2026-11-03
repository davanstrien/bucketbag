"""Tests for ``iter_keys`` — the listing + glob-filter + sort path.

Offline: the ``bucket`` fixture (``FakeBucket``) serves listings from an in-memory store, so no
bucket is ever listed over the network. This exercises the ``found.sort(...)`` guarantee, the
include-glob filter, and the non-file skip in ``_list_bucketfiles`` — none of which the
``batched_files(keys=)`` tests can reach, because passing ``keys=`` hands the keys straight
through, unsorted and unfiltered.
"""

from __future__ import annotations

from fake_bucket import FakeBucket

from bucketbag import iter_keys


def test_iter_keys_returns_sorted(bucket: FakeBucket) -> None:
    # The listing arrives in reverse lexical order; iter_keys must return the keys sorted.
    bucket.listing_order = "reversed"
    bucket.seed("ns/bucket", ["b/2.txt", "a/1.txt", "a/10.txt", "a/2.txt"])
    assert list(iter_keys("ns/bucket")) == ["a/1.txt", "a/10.txt", "a/2.txt", "b/2.txt"]


def test_iter_keys_applies_include_glob(bucket: FakeBucket) -> None:
    bucket.seed("ns/bucket", ["a/1.jp2", "a/2.txt", "a/3.jp2"])
    assert list(iter_keys("ns/bucket", include="**/*.jp2")) == ["a/1.jp2", "a/3.jp2"]


def test_iter_keys_skips_non_files(bucket: FakeBucket) -> None:
    # The listing also yields a BucketFolder for "a"; it must be dropped before sorting/yielding.
    bucket.folders = True
    bucket.seed("ns/bucket", ["a/1.txt"])
    assert list(iter_keys("ns/bucket")) == ["a/1.txt"]


def test_iter_keys_objects_yields_bucketfile(bucket: FakeBucket) -> None:
    # objects=True yields the BucketFile objects (with .size) instead of key strings, so they
    # can be fed to batched_files(keys=..., max_bytes=...) where sizes are required.
    bucket.put("ns/bucket", "a/1.jp2", data=b"x" * 7)
    bucket.put("ns/bucket", "a/2.jp2", data=b"x" * 9)
    out = list(iter_keys("ns/bucket", include="**/*.jp2", objects=True))
    assert [type(f).__name__ for f in out] == ["BucketFile", "BucketFile"]
    assert [(f.path, f.size) for f in out] == [("a/1.jp2", 7), ("a/2.jp2", 9)]
