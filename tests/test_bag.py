"""Tests for the ``Bag`` plan (P0: local compute only).

Fully offline: listing (``HfApi.list_bucket_tree``), download (``download_bucket_files``) and
write (``batch_bucket_files``) are all monkeypatched. The fake bucket store maps bucket_id ->
list[BucketFile], so source and output buckets can be listed independently (resume tests).
"""

from __future__ import annotations

import pytest
from conftest import bf, make_fake_download
from huggingface_hub import HfApi

from bucketbag import Bag, BagStats


@pytest.fixture
def fake_hub(monkeypatch):
    """A fake Hub: per-bucket listings, materializing downloads, captured writes.

    Returns a dict with ``listings`` (bucket_id -> list[BucketFile], mutable) and ``writes``
    (list of (bucket_id, add-pairs) captured from batch_bucket_files).
    """
    state = {"listings": {}, "writes": [], "list_calls": 0}

    def fake_list(self, bucket_id, prefix=None, recursive=None, token=None):  # noqa: ANN001
        state["list_calls"] += 1
        yield from state["listings"].get(bucket_id, [])

    def fake_batch(self, bucket_id, *, add=None, token=None, **kwargs):  # noqa: ANN001
        state["writes"].append((bucket_id, list(add or [])))

    monkeypatch.setattr(HfApi, "list_bucket_tree", fake_list)
    monkeypatch.setattr(HfApi, "download_bucket_files", make_fake_download())
    monkeypatch.setattr(HfApi, "batch_bucket_files", fake_batch)
    return state


def _upper_md(items):
    """The canonical fn shape: one output per item, key = source key + suffix."""
    return [(it.key + ".md", it.bytes.decode().upper()) for it in items]


def test_chain_is_lazy_and_immutable(fake_hub):
    base = Bag.from_bucket("ns/src", include="**/*.bin")
    mapped = base.map_batches(_upper_md, batch_size=2)
    final = mapped.to_bucket("ns/out")
    assert base.fn is None and mapped.out is None and final.out == "ns/out"
    assert fake_hub["list_calls"] == 0  # nothing touched the "network" yet


def test_compute_end_to_end(fake_hub, tmp_path):
    fake_hub["listings"]["ns/src"] = [bf(f"pages/p{i}.bin", size=4) for i in range(5)]
    stats = (
        Bag.from_bucket("ns/src", include="**/*.bin")
        .map_batches(_upper_md, batch_size=2)
        .to_bucket("ns/out")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert isinstance(stats, BagStats)
    assert (stats.processed, stats.skipped, stats.batches, stats.outputs) == (5, 0, 3, 5)
    # one put_files call per batch, to the right bucket, with derived keys + content
    assert [w[0] for w in fake_hub["writes"]] == ["ns/out"] * 3
    all_pairs = [p for _, pairs in fake_hub["writes"] for p in pairs]
    assert all_pairs[0] == (b"XXXX", "pages/p0.bin.md")
    assert len(all_pairs) == 5


def test_compute_resumes_by_output_existence(fake_hub, tmp_path):
    fake_hub["listings"]["ns/src"] = [bf(f"pages/p{i}.bin", size=4) for i in range(5)]
    # outputs for p0 and p3 already exist (keys prefixed by the source key)
    fake_hub["listings"]["ns/out"] = [bf("pages/p0.bin.md"), bf("pages/p3.bin.md")]
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=10)
        .to_bucket("ns/out")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert (stats.processed, stats.skipped) == (3, 2)
    done_keys = {k for _, pairs in fake_hub["writes"] for _, k in pairs}
    assert done_keys == {"pages/p1.bin.md", "pages/p2.bin.md", "pages/p4.bin.md"}


def test_resume_prefix_match_is_not_a_substring_match(fake_hub, tmp_path):
    # p1.bin.md must NOT mark p1.bin2 done; p10.bin outputs must not mark p1.bin done.
    fake_hub["listings"]["ns/src"] = [bf("p1.bin", size=4), bf("p10.bin", size=4)]
    fake_hub["listings"]["ns/out"] = [bf("p10.bin.md")]
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=10)
        .to_bucket("ns/out")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert (stats.processed, stats.skipped) == (1, 1)


def test_resume_with_embedded_out_prefix(fake_hub, tmp_path):
    # to_bucket("ns/out/run1"): put_files writes "run1/<key>.md", and listings return FULL
    # bucket paths — resume must strip the embedded prefix before matching source keys.
    # (Regression: first real smoke run reprocessed everything because it didn't.)
    fake_hub["listings"]["ns/src"] = [bf(f"pages/p{i}.bin", size=4) for i in range(3)]
    fake_hub["listings"]["ns/out"] = [bf("run1/pages/p0.bin.md"), bf("other/pages/p1.bin.md")]
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=10)
        .to_bucket("ns/out/run1")
        .compute(prefetch=0, dir=tmp_path)
    )
    # p0 done under run1/; p1's output lives under a DIFFERENT prefix -> not done
    assert (stats.processed, stats.skipped) == (2, 1)
    written = {k for _, pairs in fake_hub["writes"] for _, k in pairs}
    assert written == {"run1/pages/p1.bin.md", "run1/pages/p2.bin.md"}


def test_compute_resume_false_reprocesses(fake_hub, tmp_path):
    fake_hub["listings"]["ns/src"] = [bf("a.bin", size=4)]
    fake_hub["listings"]["ns/out"] = [bf("a.bin.md")]
    stats = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md)
        .to_bucket("ns/out")
        .compute(resume=False, prefetch=0, dir=tmp_path)
    )
    assert (stats.processed, stats.skipped) == (1, 0)


def test_setup_runs_once_and_ctx_is_passed(fake_hub, tmp_path):
    fake_hub["listings"]["ns/src"] = [bf(f"f{i}.bin", size=2) for i in range(4)]
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
        .to_bucket("ns/out")
        .compute(prefetch=0, dir=tmp_path)
    )
    assert stats.batches == 4
    assert setup_calls == [1]  # once per compute, not per batch


def test_take_returns_outputs_and_writes_nothing(fake_hub, tmp_path):
    fake_hub["listings"]["ns/src"] = [bf(f"f{i}.bin", size=2) for i in range(10)]
    out = (
        Bag.from_bucket("ns/src")
        .map_batches(_upper_md, batch_size=5)
        .to_bucket("ns/out")
        .take(3, dir=tmp_path)
    )
    assert [k for k, _ in out] == ["f0.bin.md", "f1.bin.md", "f2.bin.md"]
    assert fake_hub["writes"] == []


def test_compute_requires_fn_and_out(fake_hub):
    with pytest.raises(ValueError, match="map_batches"):
        Bag.from_bucket("ns/src").compute()
    with pytest.raises(ValueError, match="to_bucket"):
        Bag.from_bucket("ns/src").map_batches(_upper_md).compute()
