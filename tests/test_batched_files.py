"""Tests for ``batched_files`` — focusing on the **cleanup guarantee**.

All offline: ``HfApi.download_bucket_files`` is replaced (via the ``fake_download`` fixture /
``make_fake_download``) so files are materialized on local disk. Buckets are never listed because
every call passes explicit ``keys=``.

The cleanup guarantee is the library's load-bearing claim, so it's tested across
``prefetch in {0, 1, 2}`` on (a) normal completion, (b) a consumer exception, and (c) a download
failure. The prefetch>0 exception/failure paths currently **leak** temp dirs — that is issue #2,
reproduced here as strict ``xfail`` so the suite stays green while pinning the bug.
"""

from __future__ import annotations

import logging

import pytest
from conftest import bfiles, count_bb_dirs, make_fake_download
from huggingface_hub import HfApi
from huggingface_hub._buckets import BucketFile

from bucketbag import batched_files


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Reproduces #2: with prefetch>0 the consumer's exception only triggers cleanup of the "
        "currently-yielded batch; batches already downloaded in the lookahead deque are abandoned. "
        "When #2 is fixed (drain+rmtree the whole futures deque in a finally), this becomes xpass "
        "and strict=True will flag it so the marker is removed."
    ),
)
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Reproduces #2 (download-failure variant): the failing batch raises on .result(), but the "
        "already-completed lookahead batches in the deque are never rmtree'd. Fixed alongside #2."
    ),
)
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


def test_string_keys_warn_and_ignore_max_bytes(fake_download, tmp_path, caplog):
    # string keys -> sizes unknown -> max_bytes is ignored (with a warning) and only `n` binds.
    keys = [f"k{i}" for i in range(4)]
    with caplog.at_level(logging.WARNING, logger="bucketbag"):
        batches = list(
            batched_files("ns/bucket", keys=keys, n=2, max_bytes=1, prefetch=0, dir=tmp_path)
        )
    assert [[it.key for it in b] for b in batches] == [["k0", "k1"], ["k2", "k3"]]
    assert any("max_bytes ignored" in r.message for r in caplog.records)
    assert count_bb_dirs(tmp_path) == 0
