"""Tests for the ``Bag`` plan (P0: local compute only).

Fully offline: the ``bucket`` fixture (``FakeBucket``) serves listings, downloads and writes
from one in-memory store. Source and output buckets are separate entries in that store, so the
resume tests seed outputs in ``ns/out`` and read what ``compute`` wrote back from it.
"""

from __future__ import annotations

import pytest
from fake_bucket import FakeBucket

from bucketbag import Bag, BagStats


def _upper_md(items):
    """The canonical fn shape: one output per item, key = source key + suffix."""
    return [(it.key + ".md", it.bytes.decode().upper()) for it in items]


def test_chain_is_lazy_and_immutable(bucket: FakeBucket):
    base = Bag.from_bucket("ns/src", include="**/*.bin")
    mapped = base.map_batches(_upper_md, batch_size=2)
    final = mapped.to_bucket("ns/out", done=".md")
    assert base.fn is None and mapped.out is None and final.out == "ns/out"
    assert bucket.calls["list"] == 0  # nothing touched the "network" yet


def test_compute_end_to_end(bucket: FakeBucket, tmp_path):
    bucket.seed("ns/src", [f"pages/p{i}.bin" for i in range(5)], data=b"xxxx")
    stats = (
        Bag.from_bucket("ns/src", include="**/*.bin")
        .map_batches(_upper_md, batch_size=2)
        .to_bucket("ns/out", done=".md")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert isinstance(stats, BagStats)
    assert (stats.processed, stats.skipped, stats.batches, stats.outputs) == (5, 0, 3, 5)
    # one put_files call per batch, to the right bucket, with derived keys + content
    assert [bucket_id for bucket_id, _ in bucket.uploads] == ["ns/out"] * 3
    written = bucket.uploaded_keys("ns/out")
    assert written[0] == "pages/p0.bin.md"
    assert bucket.get("ns/out", "pages/p0.bin.md") == b"XXXX"
    assert len(written) == 5


def test_compute_resumes_by_output_existence(bucket: FakeBucket, tmp_path):
    bucket.seed("ns/src", [f"pages/p{i}.bin" for i in range(5)], data=b"xxxx")
    # outputs for p0 and p3 already exist (source key + the done suffix)
    bucket.seed("ns/out", ["pages/p0.bin.md", "pages/p3.bin.md"])
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=10)
        .to_bucket("ns/out", done=".md")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert (stats.processed, stats.skipped) == (3, 2)
    done_keys = set(bucket.uploaded_keys("ns/out"))
    assert done_keys == {"pages/p1.bin.md", "pages/p2.bin.md", "pages/p4.bin.md"}


def test_resume_matches_the_exact_done_output(bucket: FakeBucket, tmp_path):
    # Only p1.bin + ".md" marks p1.bin done; p10.bin's output does not.
    bucket.seed("ns/src", ["p1.bin", "p10.bin"], data=b"xxxx")
    bucket.seed("ns/out", ["p10.bin.md"])
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=10)
        .to_bucket("ns/out", done=".md")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert (stats.processed, stats.skipped) == (1, 1)


def test_resume_with_embedded_out_prefix(bucket: FakeBucket, tmp_path):
    # to_bucket("ns/out/run1"): put_files writes "run1/<key>.md", and listings return FULL
    # bucket paths — resume must strip the embedded prefix before matching source keys.
    # (Regression: first real smoke run reprocessed everything because it didn't.)
    bucket.seed("ns/src", [f"pages/p{i}.bin" for i in range(3)], data=b"xxxx")
    # Outputs of other runs that the Hub's string-prefix listing for "run1" also returns.
    # "run1-pages/p1.bin.md" is adversarial: cutting len("run1/") characters off it leaves
    # "pages/p1.bin.md", so only bucketbag's own "run1/" filter keeps it from marking p1 done.
    bucket.seed("ns/out", ["run1/pages/p0.bin.md", "run10/pages/p1.bin.md", "run1-pages/p1.bin.md"])
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=10)
        .to_bucket("ns/out/run1", done=".md")
        .compute(prefetch=0, dir=tmp_path)
    )
    # p0 done under run1/; p1's outputs live under sibling runs -> not done
    assert (stats.processed, stats.skipped) == (2, 1)
    written = set(bucket.uploaded_keys("ns/out"))
    assert written == {"run1/pages/p1.bin.md", "run1/pages/p2.bin.md"}


def test_compute_resume_false_reprocesses(bucket: FakeBucket, tmp_path):
    bucket.seed("ns/src", ["a.bin"], data=b"xxxx")
    bucket.seed("ns/out", ["a.bin.md"])
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md)
        .to_bucket("ns/out", done=".md")
        .compute(resume=False, prefetch=0, dir=tmp_path)
    )
    assert (stats.processed, stats.skipped) == (1, 0)


def test_setup_runs_once_and_ctx_is_passed(bucket: FakeBucket, tmp_path):
    bucket.seed("ns/src", [f"f{i}.bin" for i in range(4)], data=b"xx")
    setup_calls = []

    def load_model():
        setup_calls.append(1)
        return {"model": "m"}

    def fn(items, ctx):
        assert ctx == {"model": "m"}
        return [(it.key + ".out", "y") for it in items]

    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(fn, batch_size=1, setup=load_model)
        .to_bucket("ns/out", done=".out")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert stats.batches == 4
    assert setup_calls == [1]  # once per compute, not per batch


def test_take_returns_outputs_and_writes_nothing(bucket: FakeBucket, tmp_path):
    bucket.seed("ns/src", [f"f{i}.bin" for i in range(10)], data=b"xx")
    out = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=5)
        .to_bucket("ns/out", done=".md")
        .take(3, dir=tmp_path)
    )
    assert [k for k, _ in out] == ["f0.bin.md", "f1.bin.md", "f2.bin.md"]
    assert bucket.calls["upload"] == 0


def test_compute_requires_fn_and_out(bucket: FakeBucket):
    with pytest.raises(ValueError, match="map_batches"):
        Bag.from_bucket("ns/src").compute()
    with pytest.raises(ValueError, match="to_bucket"):
        Bag.from_bucket("ns/src").map_batches(_upper_md).compute()
