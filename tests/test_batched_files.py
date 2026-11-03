"""Tests for ``batched_files`` — focusing on the **cleanup guarantee**.

All offline: the ``bucket`` fixture (``FakeBucket``) holds the files, and its downloads write
their bytes to local disk. Every call passes explicit ``keys=``, so ``batched_files`` never
lists; only the resume composition test lists, through ``iter_keys``.

The cleanup guarantee is the library's load-bearing claim, so it's tested across
``prefetch in {0, 1, 2}`` on (a) normal completion, (b) a consumer exception, and (c) a download
failure. The prefetch>0 exception and failure tests pin issue #2: temp dirs of lookahead
batches leaked when the loop stopped early.
"""

from __future__ import annotations

import pytest
from conftest import count_bb_dirs
from fake_bucket import FakeBucket, FaultInjected
from huggingface_hub import BucketFile

from bucketbag import batched_files, iter_keys

BUCKET = "ns/bucket"


def seed_files(bucket: FakeBucket, n: int, size: int) -> list[BucketFile]:
    """Store ``n`` files ``dir/f0.bin`` .. ``dir/f<n-1>.bin`` in ``BUCKET``, each ``size`` bytes."""
    names = [f"dir/f{i}.bin" for i in range(n)]
    return bucket.seed(BUCKET, names, data=b"x" * size)


# --------------------------------------------------------------------------- #
# Normal completion: every prefetch level must leave zero temp dirs.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_cleans_up_on_normal_completion(bucket: FakeBucket, tmp_path, prefetch):
    keys = seed_files(bucket, 9, size=4)
    for batch in batched_files(BUCKET, keys=keys, n=1, prefetch=prefetch, dir=tmp_path):
        assert len(batch) == 1
        assert batch[0].bytes == b"x" * 4
    assert count_bb_dirs(tmp_path) == 0


# --------------------------------------------------------------------------- #
# Consumer exception.
# --------------------------------------------------------------------------- #
def _raise_after(make_gen, at):
    """Consume the generator from ``make_gen()``, raising after the ``at``-th batch.

    The generator is the loop's only reference, so when the frame unwinds on the raise the
    generator is closed (GeneratorExit) and its per-batch ``finally`` cleanup runs — exactly
    the real-world lifetime of a generator consumed inside a function that then exits.
    """
    for i, _batch in enumerate(make_gen()):
        if i == at:
            raise RuntimeError("boom")


def test_cleans_up_on_consumer_exception_prefetch_0(bucket: FakeBucket, tmp_path):
    keys = seed_files(bucket, 9, size=4)
    with pytest.raises(RuntimeError, match="boom"):
        _raise_after(lambda: batched_files(BUCKET, keys=keys, n=1, prefetch=0, dir=tmp_path), 1)
    assert count_bb_dirs(tmp_path) == 0


def test_cleans_up_on_consumer_exception_prefetch_2(bucket: FakeBucket, tmp_path):
    keys = seed_files(bucket, 9, size=4)
    with pytest.raises(RuntimeError, match="boom"):
        _raise_after(lambda: batched_files(BUCKET, keys=keys, n=1, prefetch=2, dir=tmp_path), 1)
    assert count_bb_dirs(tmp_path) == 0


# --------------------------------------------------------------------------- #
# Download failure (one bad batch mid-loop).
# --------------------------------------------------------------------------- #
def test_cleans_up_on_download_failure_prefetch_0(bucket: FakeBucket, tmp_path):
    # Batch #2 of 9 (n=1) fails. prefetch=0 -> the failing chunk cleans its own tmpdir and no
    # lookahead exists, so nothing leaks.
    bucket.fail_download_on.add(2)
    keys = seed_files(bucket, 9, size=4)
    with pytest.raises(FaultInjected):
        for _ in batched_files(BUCKET, keys=keys, n=1, prefetch=0, dir=tmp_path):
            pass
    assert count_bb_dirs(tmp_path) == 0


def test_cleans_up_on_download_failure_prefetch_2(bucket: FakeBucket, tmp_path):
    bucket.fail_download_on.add(3)
    keys = seed_files(bucket, 9, size=4)
    with pytest.raises(FaultInjected):
        for _ in batched_files(BUCKET, keys=keys, n=1, prefetch=2, dir=tmp_path):
            pass
    assert count_bb_dirs(tmp_path) == 0


# --------------------------------------------------------------------------- #
# Batching / ordering / content.
# --------------------------------------------------------------------------- #
def test_yields_batches_in_key_order(bucket: FakeBucket, tmp_path):
    keys = bucket.seed(BUCKET, [f"k{i}" for i in range(6)], data=b"x" * 3)
    seen = []
    # Read content *inside* the loop: a batch's tmpdir is deleted once the loop advances.
    for batch in batched_files(BUCKET, keys=keys, n=2, prefetch=0, dir=tmp_path):
        seen.append([it.key for it in batch])
        assert all(it.bytes == b"x" * 3 for it in batch)
    assert seen == [["k0", "k1"], ["k2", "k3"], ["k4", "k5"]]
    assert count_bb_dirs(tmp_path) == 0


def test_max_bytes_respected_with_bucketfile_keys(bucket: FakeBucket, tmp_path):
    # sizes 10,10,10,10 with max_bytes=25 -> [0,1],[2,3]
    keys = bucket.seed(BUCKET, [f"k{i}" for i in range(4)], data=b"x" * 10)
    batches = list(batched_files(BUCKET, keys=keys, max_bytes=25, prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [["k0", "k1"], ["k2", "k3"]]


def test_empty_keys_yields_nothing(bucket: FakeBucket, tmp_path):
    assert list(batched_files(BUCKET, keys=[], n=5, dir=tmp_path)) == []
    assert count_bb_dirs(tmp_path) == 0


# --------------------------------------------------------------------------- #
# max_bytes + string keys: the bound can't be honored, so batched_files fails fast.
# String keys carry no size. Running the loop without the bound against the default
# RAM-tmpfs scratch dir risks an OOMKill, not a disk-full, so a caller who asked for a
# bound gets a ValueError instead of an unbounded loop.
# --------------------------------------------------------------------------- #
def test_string_keys_with_max_bytes_raises(bucket: FakeBucket, tmp_path):
    keys = [f"k{i}" for i in range(4)]
    bucket.seed(BUCKET, keys)
    with pytest.raises(ValueError, match="max_bytes requires sized keys"):
        list(batched_files(BUCKET, keys=keys, max_bytes=1, prefetch=0, dir=tmp_path))
    assert count_bb_dirs(tmp_path) == 0


def test_mixed_keys_with_max_bytes_raises(bucket: FakeBucket, tmp_path):
    # A BucketFile at index 0 must not let a stray string sneak through: every key must be
    # sized when max_bytes is set (else _pack treats the unsized one as size 0).
    sized = bucket.put(BUCKET, "k0", data=b"x" * 10)
    bucket.put(BUCKET, "k1")
    keys: list[BucketFile | str] = [sized, "k1"]
    with pytest.raises(ValueError, match="max_bytes requires sized keys"):
        list(batched_files(BUCKET, keys=keys, max_bytes=100, prefetch=0, dir=tmp_path))
    assert count_bb_dirs(tmp_path) == 0


def test_string_keys_without_max_bytes_ok(bucket: FakeBucket, tmp_path):
    # Strings with a count cap only (no max_bytes) is a legitimate path and must still work.
    keys = [f"k{i}" for i in range(4)]
    bucket.seed(BUCKET, keys, data=b"x")
    batches = list(batched_files(BUCKET, keys=keys, n=2, prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [["k0", "k1"], ["k2", "k3"]]
    assert count_bb_dirs(tmp_path) == 0


def test_resume_composition_honors_max_bytes(bucket: FakeBucket, tmp_path):
    # The README "Writing + resume" shape: iter_keys(objects=True) -> filter done -> batched_files.
    # Regression test for the issue itself: max_bytes must be honored end to end, not dropped.
    SRC, OUT = "ns/src", "ns/out"
    bucket.seed(SRC, [f"{i}.jp2" for i in range(4)], data=b"x" * 10)  # 0..3, 10 bytes each
    bucket.seed(OUT, ["0.md", "1.md"], data=b"x")  # 0 and 1 already done

    done = {k.removesuffix(".md") for k in iter_keys(OUT, include="**/*.md")}
    keys = [
        k
        for k in iter_keys(SRC, include="**/*.jp2", objects=True)
        if k.path.removesuffix(".jp2") not in done
    ]
    # Resume dropped 0 and 1; what's left is sized, so max_bytes binds.
    assert [k.path for k in keys] == ["2.jp2", "3.jp2"]
    batches = list(batched_files(SRC, keys=keys, max_bytes=15, prefetch=0, dir=tmp_path))
    # Non-vacuous via the "fewer items than n" route: only 2 files vs the default n=20, so the
    # count cap can never bind — the split is purely byte-driven (10+10=20 > 15). Dropping
    # max_bytes would collapse this to one batch and fail the assertion below.
    assert [[it.key for it in b] for b in batches] == [["2.jp2"], ["3.jp2"]]
    assert count_bb_dirs(tmp_path) == 0
