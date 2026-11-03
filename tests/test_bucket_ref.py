"""Tests for ``BucketRef``: parsing bucket references and joining prefixes (CONTRACT §1)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bucketbag._core import BucketRef

pytestmark = pytest.mark.internals

WEB = "https://huggingface.co/buckets"


@pytest.mark.parametrize(
    ("raw", "bucket_id", "prefix"),
    [
        ("ns/bucket", "ns/bucket", ""),
        ("ns/bucket/", "ns/bucket", ""),
        ("ns/bucket/p", "ns/bucket", "p"),
        ("ns/bucket/a/b/", "ns/bucket", "a/b/"),  # the trailing slash is kept
        ("/ns/bucket/sp", "ns/bucket", "sp"),
        ("hf://buckets/ns/bucket", "ns/bucket", ""),
        ("hf://buckets/ns/bucket/", "ns/bucket", ""),  # never a slash on an empty prefix
        ("hf://buckets/ns/bucket/a/b", "ns/bucket", "a/b"),
        ("hf://buckets/ns/bucket/a/b/", "ns/bucket", "a/b/"),
        (f"{WEB}/ns/bucket", "ns/bucket", ""),
        (f"{WEB}/ns/bucket/tree/a/b/", "ns/bucket", "a/b/"),
        (f"{WEB}/ns/bucket/resolve/a/b", "ns/bucket", "a/b"),
        (f"{WEB}/ns/bucket/tree/a/b/?q=1", "ns/bucket", "a/b/"),  # a query keeps the slash
        (f"{WEB}/ns/bucket/tree/a/b?q=/", "ns/bucket", "a/b"),  # and does not invent one
    ],
)
def test_parse(raw: str, bucket_id: str, prefix: str) -> None:
    assert BucketRef.parse(raw) == BucketRef(bucket_id, prefix)


@pytest.mark.parametrize("bad", ["", "ns", "x", "/", "hf://datasets/ns/name"])
def test_parse_refuses_what_is_not_a_bucket(bad: str) -> None:
    with pytest.raises(ValueError):
        BucketRef.parse(bad)


@pytest.mark.parametrize(
    ("prefix", "joined", "expected"),
    [
        ("", "a/b/", "a/b/"),
        ("out", "part-", "out/part-"),
        ("out/", "part-", "out/part-"),
        ("out/", None, "out/"),
        ("out", "", "out"),
    ],
)
def test_under_joins_as_directories(prefix: str, joined: str | None, expected: str) -> None:
    assert BucketRef("ns/b", prefix).under(joined).prefix == expected


@pytest.mark.parametrize(
    ("prefix", "key", "expected"),
    [("", "k", "k"), ("out", "k", "out/k"), ("out/", "k", "out/k"), ("a/b/", "c/d", "a/b/c/d")],
)
def test_key_for_never_doubles_a_slash(prefix: str, key: str, expected: str) -> None:
    assert BucketRef("ns/b", prefix).key_for(key) == expected


# --- property ---------------------------------------------------------------
# Names use the characters a Hub id allows, so parse_hf_uri accepts the hf:// form too.
_NAME = st.text(
    min_size=1, max_size=8, alphabet=st.characters(whitelist_categories=("Ll", "Nd"))
).filter(lambda s: not s.startswith(("-", ".")))
_SEG = st.text(min_size=1, max_size=4, alphabet="abcxyz019-_.")


@given(
    ns=_NAME,
    name=_NAME,
    extra=st.lists(_SEG, max_size=4),
    scheme=st.sampled_from(["", "hf://buckets/"]),
)
def test_parse_roundtrip(ns: str, name: str, extra: list[str], scheme: str) -> None:
    ref = BucketRef.parse(scheme + "/".join([ns, name, *extra]))
    assert ref == BucketRef(f"{ns}/{name}", "/".join(extra))
