"""Tests for ``bucketbag._pack`` — the batch-size accounting.

These are the load-bearing invariants of the batching logic. ``_pack`` is pure and trivially
deterministic, so property-based tests (Hypothesis) cover the byte/count cap interactions far
more thoroughly than hand-written cases.
"""

from __future__ import annotations

from conftest import bf
from hypothesis import given
from hypothesis import strategies as st

from bucketbag import _pack


def paths(batches):
    return [[f.path for f in b] for b in batches]


# --- concrete cases ---------------------------------------------------------
def test_pack_count_only():
    items = [bf(f"f{i}.bin", size=1) for i in range(7)]
    assert paths(_pack(items, n=3, max_bytes=None)) == [
        ["f0.bin", "f1.bin", "f2.bin"],
        ["f3.bin", "f4.bin", "f5.bin"],
        ["f6.bin"],
    ]


def test_pack_bytes_only():
    items = [bf("a", size=10), bf("b", size=10), bf("c", size=10), bf("d", size=10)]
    # cap 25 -> [a,b]=20 (next would be 30>25), [c,d]=20
    assert paths(_pack(items, n=None, max_bytes=25)) == [["a", "b"], ["c", "d"]]


def test_pack_single_file_larger_than_max_bytes_is_own_batch():
    items = [bf("a", size=10), bf("huge", size=1000), bf("b", size=10)]
    batches = list(_pack(items, n=None, max_bytes=50))
    assert paths(batches) == [["a"], ["huge"], ["b"]]
    # the oversized batch legitimately exceeds the cap — files can't be split.
    assert sum(f.size for f in batches[1]) > 50


def test_pack_no_caps_is_one_batch():
    items = [bf(f"f{i}.bin", size=1) for i in range(5)]
    assert len(list(_pack(items, n=None, max_bytes=None))) == 1


def test_pack_empty():
    assert list(_pack([], n=3, max_bytes=10)) == []


def test_pack_count_and_bytes_whichever_first():
    # n=10 won't bind (only 3 files); max_bytes=20 binds at 3 files of size 7 (21 > 20).
    items = [bf("a", 7), bf("b", 7), bf("c", 7), bf("d", 7)]
    assert paths(_pack(items, n=10, max_bytes=20)) == [["a", "b"], ["c", "d"]]


def test_pack_unknown_size_counts_as_zero():
    # A plain object with no `.size` -> getattr(...,0) -> 0; never triggers the bytes cap.
    class Bare:
        def __init__(self, p):
            self.path = p

    items = [Bare(f"f{i}") for i in range(4)]
    assert len(list(_pack(items, n=None, max_bytes=1))) == 1  # all zero-size -> one batch


def test_pack_n_zero_is_degenerate_leading_empty_batch():
    # n=0 is a degenerate cap that bucketbag never produces (batched_files defaults n=20 and
    # never passes 0). _pack does not special-case it: `len(batch) >= 0` is true before the first
    # append, so a leading empty batch is emitted. Pinned here so the boundary the property tests
    # below deliberately exclude is still covered, and any future n<=0 handling is a visible diff.
    items = [bf("a", 1), bf("b", 1)]
    assert paths(_pack(items, n=0, max_bytes=None)) == [[], ["a"], ["b"]]


# --- properties -------------------------------------------------------------
# sizes of files; n (count cap); max_bytes (byte cap). Both caps optional. The count cap is
# drawn from {None} ∪ [1, 2000] — the supported domain (n=0 is degenerate, covered above).
sizes_st = st.lists(
    st.integers(min_value=0, max_value=1000),
    max_size=40,
)
caps_st = st.one_of(st.none(), st.integers(min_value=1, max_value=2000))


def _flatten(batches):
    return [f for b in batches for f in b]


@given(sizes=sizes_st, n=caps_st, max_bytes=caps_st)
def test_pack_preserves_order(sizes, n, max_bytes):
    items = [bf(f"k{i}", size=s) for i, s in enumerate(sizes)]
    out = _flatten(_pack(items, n, max_bytes))
    assert [f.path for f in out] == [f"k{i}" for i in range(len(sizes))]


@given(sizes=sizes_st, n=caps_st, max_bytes=caps_st)
def test_pack_no_empty_batches(sizes, n, max_bytes):
    batches = list(_pack([bf(f"k{i}", s) for i, s in enumerate(sizes)], n, max_bytes))
    assert all(len(b) >= 1 for b in batches)


@given(sizes=sizes_st, n=caps_st, max_bytes=caps_st)
def test_pack_count_cap_respected(sizes, n, max_bytes):
    batches = list(_pack([bf(f"k{i}", s) for i, s in enumerate(sizes)], n, max_bytes))
    if n is not None:
        assert all(len(b) <= n for b in batches)


@given(sizes=sizes_st, max_bytes=st.integers(min_value=1, max_value=2000))
def test_pack_bytes_cap_respected_except_oversize_singletons(sizes, max_bytes):
    # n is irrelevant to the bytes invariant, so leave it None.
    batches = list(_pack([bf(f"k{i}", s) for i, s in enumerate(sizes)], None, max_bytes))
    for b in batches:
        total = sum(f.size for f in b)
        if len(b) == 1:
            # a single file may exceed the cap (files can't be split) — always fine.
            continue
        assert total <= max_bytes


@given(sizes=sizes_st, max_bytes=st.integers(min_value=1, max_value=2000))
def test_pack_greedy_correctness(sizes, max_bytes):
    """Every batch is maximal: appending the first file of the *next* batch would overflow it
    (unless the batch is a single oversize file, or the next file is itself oversize)."""
    items = [bf(f"k{i}", s) for i, s in enumerate(sizes)]
    batches = list(_pack(items, None, max_bytes))
    flat = _flatten(batches)
    # walk items in order, tracking batch boundaries
    idx = 0
    for b in batches:
        nb = len(b)
        total = sum(f.size for f in b)
        # if there's a following item, adding it should have overflowed (count == already packed).
        if idx + nb < len(items) and len(b) >= 1:
            nxt = flat[idx + nb].size
            # Batch closes only because adding `next` would exceed max_bytes,
            # unless this batch is a forced oversize singleton.
            forced_singleton = len(b) == 1 and total > max_bytes
            assert forced_singleton or total + nxt > max_bytes
        idx += nb
