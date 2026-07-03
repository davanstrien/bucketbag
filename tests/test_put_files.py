"""Tests for the per-object write helpers (``put_files`` / ``put_bytes`` / ``put_text``).

Fully offline: ``HfApi.batch_bucket_files`` is monkeypatched to capture what would be uploaded.
"""

from __future__ import annotations

import pytest
from huggingface_hub import HfApi

from bucketbag import put_bytes, put_files, put_text


@pytest.fixture
def capture_batch(monkeypatch):
    """Record every ``batch_bucket_files`` call as ``(bucket_id, add)``."""
    calls: list[tuple[str, list[tuple[bytes, str]]]] = []

    def fake_batch(self, bucket_id, *, add=None, token=None, **kwargs):  # noqa: ANN001
        calls.append((bucket_id, list(add or [])))

    monkeypatch.setattr(HfApi, "batch_bucket_files", fake_batch)
    return calls


def test_put_files_one_call_many_objects(capture_batch):
    put_files([("a.md", "alpha"), ("b.md", b"beta")], "ns/bucket")
    assert len(capture_batch) == 1
    bucket_id, add = capture_batch[0]
    assert bucket_id == "ns/bucket"
    assert add == [(b"alpha", "a.md"), (b"beta", "b.md")]


def test_put_files_joins_embedded_prefix(capture_batch):
    put_files([("x/y.md", "t")], "hf://buckets/ns/bucket/out/run1")
    bucket_id, add = capture_batch[0]
    assert bucket_id == "ns/bucket"
    assert add == [(b"t", "out/run1/x/y.md")]


def test_put_files_empty_is_noop(capture_batch):
    put_files([], "ns/bucket")
    assert capture_batch == []


def test_put_files_encoding(capture_batch):
    put_files([("k", "café")], "ns/bucket", encoding="latin-1")
    assert capture_batch[0][1] == [("café".encode("latin-1"), "k")]


def test_put_bytes_single_object(capture_batch):
    put_bytes(b"\x00\x01", "ns/bucket/prefix", "blob.bin")
    bucket_id, add = capture_batch[0]
    assert bucket_id == "ns/bucket"
    assert add == [(b"\x00\x01", "prefix/blob.bin")]


def test_put_text_encodes(capture_batch):
    put_text("héllo", "ns/bucket", "page.md")
    assert capture_batch[0][1] == [("héllo".encode(), "page.md")]
