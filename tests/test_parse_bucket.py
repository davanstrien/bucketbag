"""Tests for ``bucketbag._parse_bucket``."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bucketbag import _parse_bucket


# --- concrete cases ---------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ns/bucket", ("ns/bucket", "")),
        ("ns/bucket/", ("ns/bucket", "")),  # trailing slash stripped
        ("ns/bucket/p", ("ns/bucket", "p")),
        ("ns/bucket/some/deep/prefix", ("ns/bucket", "some/deep/prefix")),
        ("hf://buckets/ns/bucket", ("ns/bucket", "")),
        ("hf://buckets/ns/bucket/p", ("ns/bucket", "p")),
        ("hf://buckets/ns/bucket/a/b/c", ("ns/bucket", "a/b/c")),
        ("/ns/bucket/sp", ("ns/bucket", "sp")),  # leading slash stripped
    ],
)
def test_parse_bucket_cases(raw, expected):
    assert _parse_bucket(raw) == expected


@pytest.mark.parametrize("bad", ["", "ns", "x", "/"])
def test_parse_bucket_too_short_raises(bad):
    with pytest.raises(ValueError):
        _parse_bucket(bad)


# --- property ---------------------------------------------------------------
# A valid bucket id is exactly two path components; the prefix is everything after.
# Alphabets exclude `/` and whitespace so components split cleanly.
_NAME = st.text(
    min_size=1,
    max_size=8,
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="/ "),
)
_SEG = st.text(
    min_size=1,
    max_size=4,
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="/"),
)


@given(
    ns=_NAME,
    name=_NAME,
    extra=st.lists(_SEG, max_size=4),
    scheme=st.sampled_from(["", "hf://buckets/"]),
)
def test_parse_bucket_roundtrip(ns, name, extra, scheme):
    raw = scheme + "/".join([ns, name, *extra])
    bucket_id, prefix = _parse_bucket(raw)
    assert bucket_id == f"{ns}/{name}"
    assert prefix == "/".join(extra)
    # bucket_id is always exactly "ns/name" — two components, one slash.
    assert bucket_id.count("/") == 1
