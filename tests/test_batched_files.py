"""Tests for ``batched_files`` — focusing on the **cleanup guarantee**.

All offline: ``HfApi.download_bucket_files`` is replaced (via the ``fake_download`` fixture /
``make_fake_download``) so files are materialized on local disk. Buckets are never listed because
every call passes explicit ``keys=``.

The cleanup guarantee is the library's load-bearing claim, so it's tested across
``prefetch in {0, 1, 2}`` on (a) normal completion, (b) a consumer exception, and (c) a download
failure. The prefetch>0 exception/failure paths used to leak temp dirs (issue #2, fixed by
draining the lookahead deque in a ``finally``); the former strict-xfail repros now run as
regular regression tests.
"""

from __future__ import annotations

import pytest
from conftest import bf, bfiles, count_bb_dirs, make_fake_download
from huggingface_hub import BucketFile, HfApi

from bucketbag import batched_files, iter_keys


# --------------------------------------------------------------------------- #
# Normal completion: every prefetch level must leave zero temp dirs.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_cleans_up_on_normal_completion(fake_download, tmp_path, prefetch):
    keys = bfiles(9, size=4)
    for batch in batched_files("ns/bucket", keys=keys, n=1, prefetch=prefetch, dir=tmp_path):
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


def test_cleans_up_on_consumer_exception_prefetch_0(fake_download, tmp_path):
    keys = bfiles(9, size=4)
    with pytest.raises(RuntimeError, match="boom"):
        _raise_after(
            lambda: batched_files("ns/bucket", keys=keys, n=1, prefetch=0, dir=tmp_path), 1
        )
    assert count_bb_dirs(tmp_path) == 0


def test_cleans_up_on_consumer_exception_prefetch_2(fake_download, tmp_path):
    keys = bfiles(9, size=4)
    with pytest.raises(RuntimeError, match="boom"):
        _raise_after(
            lambda: batched_files("ns/bucket", keys=keys, n=1, prefetch=2, dir=tmp_path), 1
        )
    assert count_bb_dirs(tmp_path) == 0


# --------------------------------------------------------------------------- #
# Download failure (one bad batch mid-loop).
# --------------------------------------------------------------------------- #
def test_cleans_up_on_download_failure_prefetch_0(monkeypatch, tmp_path):
    # Batch #2 of 9 (n=1) fails. prefetch=0 -> the failing chunk cleans its own tmpdir and no
    # lookahead exists, so nothing leaks.
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download(fail_on_call=2))
    keys = bfiles(9, size=4)
    with pytest.raises(RuntimeError, match="simulated download failure"):
        for _ in batched_files("ns/bucket", keys=keys, n=1, prefetch=0, dir=tmp_path):
            pass
    assert count_bb_dirs(tmp_path) == 0


def test_cleans_up_on_download_failure_prefetch_2(monkeypatch, tmp_path):
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download(fail_on_call=3))
    keys = bfiles(9, size=4)
    with pytest.raises(RuntimeError, match="simulated download failure"):
        for _ in batched_files("ns/bucket", keys=keys, n=1, prefetch=2, dir=tmp_path):
            pass
    assert count_bb_dirs(tmp_path) == 0


# --------------------------------------------------------------------------- #
# Batching / ordering / content.
# --------------------------------------------------------------------------- #
def test_yields_batches_in_key_order(fake_download, tmp_path):
    keys = [BucketFile(type="file", path=f"k{i}", size=3, xetHash="") for i in range(6)]
    seen = []
    # Read content *inside* the loop: a batch's tmpdir is deleted once the loop advances.
    for batch in batched_files("ns/bucket", keys=keys, n=2, prefetch=0, dir=tmp_path):
        seen.append([it.key for it in batch])
        assert all(it.bytes == b"x" * 3 for it in batch)
    assert seen == [["k0", "k1"], ["k2", "k3"], ["k4", "k5"]]
    assert count_bb_dirs(tmp_path) == 0


def test_max_bytes_respected_with_bucketfile_keys(fake_download, tmp_path):
    # sizes 10,10,10,10 with max_bytes=25 -> [0,1],[2,3]
    keys = [BucketFile(type="file", path=f"k{i}", size=10, xetHash="") for i in range(4)]
    batches = list(batched_files("ns/bucket", keys=keys, max_bytes=25, prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [["k0", "k1"], ["k2", "k3"]]


def test_empty_keys_yields_nothing(fake_download, tmp_path):
    assert list(batched_files("ns/bucket", keys=[], n=5, dir=tmp_path)) == []
    assert count_bb_dirs(tmp_path) == 0


# --------------------------------------------------------------------------- #
# max_bytes + string keys: the bound can't be honored, so we fail fast.
# String keys carry no size; the old warn-and-drop behaviour ran the loop
# unbounded — against the default RAM-tmpfs scratch dir that is an OOMKill, not
# a disk-full. A caller who asked for a bound should not get an unbounded loop.
# --------------------------------------------------------------------------- #
def test_string_keys_with_max_bytes_raises(fake_download, tmp_path):
    keys = [f"k{i}" for i in range(4)]
    with pytest.raises(ValueError, match="max_bytes requires sized keys"):
        list(batched_files("ns/bucket", keys=keys, max_bytes=1, prefetch=0, dir=tmp_path))
    assert count_bb_dirs(tmp_path) == 0


def test_mixed_keys_with_max_bytes_raises(fake_download, tmp_path):
    # A BucketFile at index 0 must not let a stray string sneak through: every key must be
    # sized when max_bytes is set (else _pack treats the unsized one as size 0).
    keys = [BucketFile(type="file", path="k0", size=10, xetHash=""), "k1"]
    with pytest.raises(ValueError, match="max_bytes requires sized keys"):
        list(batched_files("ns/bucket", keys=keys, max_bytes=100, prefetch=0, dir=tmp_path))
    assert count_bb_dirs(tmp_path) == 0


def test_string_keys_without_max_bytes_ok(fake_download, tmp_path):
    # Strings with a count cap only (no max_bytes) is a legitimate path and must still work.
    keys = [f"k{i}" for i in range(4)]
    batches = list(batched_files("ns/bucket", keys=keys, n=2, prefetch=0, dir=tmp_path))
    assert [[it.key for it in b] for b in batches] == [["k0", "k1"], ["k2", "k3"]]
    assert count_bb_dirs(tmp_path) == 0


def test_resume_composition_honors_max_bytes(monkeypatch, tmp_path):
    # The README "Writing + resume" shape: iter_keys(objects=True) -> filter done -> batched_files.
    # Regression test for the issue itself: max_bytes must be honored end to end, not dropped.
    SRC, OUT = "ns/src", "ns/out"

    def fake_list(self, bucket_id, *, prefix=None, recursive=False, **kwargs):  # noqa: ANN001
        if bucket_id == SRC:
            return [bf(f"{i}.jp2", size=10) for i in range(4)]  # 0..3, 10 bytes each
        if bucket_id == OUT:
            return [bf("0.md", size=1), bf("1.md", size=1)]  # 0 and 1 already done
        return []

    monkeypatch.setattr(HfApi, "list_bucket_tree", fake_list)
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download())

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
