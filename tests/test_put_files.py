"""Tests for the per-object write helpers (``put_files`` / ``put_bytes`` / ``put_text``).

Fully offline: the ``bucket`` fixture (``FakeBucket``) stores every upload. ``bucket.uploads``
records each ``batch_bucket_files`` call as ``(bucket_id, keys in call order)``, and
``bucket.get`` returns the stored bytes.
"""

from __future__ import annotations

from fake_bucket import FakeBucket

from bucketbag import put_bytes, put_files, put_text


def test_put_files_one_call_many_objects(bucket: FakeBucket) -> None:
    put_files([("a.md", "alpha"), ("b.md", b"beta")], "ns/bucket")
    assert bucket.uploads == [("ns/bucket", ["a.md", "b.md"])]
    assert bucket.get("ns/bucket", "a.md") == b"alpha"
    assert bucket.get("ns/bucket", "b.md") == b"beta"


def test_put_files_joins_embedded_prefix(bucket: FakeBucket) -> None:
    put_files([("x/y.md", "t")], "hf://buckets/ns/bucket/out/run1")
    assert bucket.uploads[0] == ("ns/bucket", ["out/run1/x/y.md"])
    assert bucket.get("ns/bucket", "out/run1/x/y.md") == b"t"


def test_put_files_empty_is_noop(bucket: FakeBucket) -> None:
    put_files([], "ns/bucket")
    assert bucket.calls["upload"] == 0


def test_put_files_encoding(bucket: FakeBucket) -> None:
    put_files([("k", "café")], "ns/bucket", encoding="latin-1")
    assert bucket.uploads[0][1] == ["k"]
    assert bucket.get("ns/bucket", "k") == "café".encode("latin-1")


def test_put_bytes_single_object(bucket: FakeBucket) -> None:
    put_bytes(b"\x00\x01", "ns/bucket/prefix", "blob.bin")
    assert bucket.uploads[0] == ("ns/bucket", ["prefix/blob.bin"])
    assert bucket.get("ns/bucket", "prefix/blob.bin") == b"\x00\x01"


def test_put_text_encodes(bucket: FakeBucket) -> None:
    put_text("héllo", "ns/bucket", "page.md")
    assert bucket.uploads[0][1] == ["page.md"]
    assert bucket.get("ns/bucket", "page.md") == "héllo".encode()
