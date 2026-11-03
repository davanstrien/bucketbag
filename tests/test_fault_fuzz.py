"""Fault fuzzing for ``Bag.compute``: whatever mix of failures a run meets, it leaves no
temporary files, and a re-run against a healthy store writes each missing output exactly once.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest
from conftest import count_bb_dirs
from fake_bucket import FakeBucket, FaultInjected
from hypothesis import given, settings
from hypothesis import strategies as st

from bucketbag import Bag, LoadedItem

SRC, OUT = "ns/src", "ns/out"

# What can go wrong in a run, and the 1-based call (or file) it hits.
FAULTS = st.dictionaries(
    keys=st.sampled_from(["map", "download", "upload", "partial upload", "missing"]),
    values=st.integers(min_value=1, max_value=4),
)


class Boom(RuntimeError):
    pass


class MapFunction:
    """Uppercases each page into ``<key>.md``; raises on call ``fail_on`` while ``failing``."""

    def __init__(self, fail_on: int | None) -> None:
        self.fail_on = fail_on
        self.failing = True
        self.calls = 0

    def __call__(self, items: list[LoadedItem]) -> list[tuple[str, bytes]]:
        self.calls += 1
        if self.failing and self.calls == self.fail_on:
            raise Boom(f"map call {self.calls}")
        return [(it.key + ".md", it.bytes.upper()) for it in items]


def inject(bucket: FakeBucket, faults: dict[str, int], keys: list[str]) -> None:
    if "download" in faults:
        bucket.fail_download_on.add(faults["download"])
    if "upload" in faults:
        bucket.fail_upload_on.add(faults["upload"])
    if "partial upload" in faults:
        bucket.partial_upload_on.add(faults["partial upload"])
    if "missing" in faults and keys:
        bucket.missing.add(keys[(faults["missing"] - 1) % len(keys)])


def heal(bucket: FakeBucket, fn: MapFunction) -> None:
    bucket.fail_download_on.clear()
    bucket.fail_upload_on.clear()
    bucket.partial_upload_on.clear()
    bucket.missing.clear()
    fn.failing = False


@settings(max_examples=200, deadline=None)
@given(
    n_files=st.integers(min_value=0, max_value=12),
    batch_size=st.integers(min_value=1, max_value=4),
    prefetch=st.integers(min_value=0, max_value=3),
    faults=FAULTS,
)
def test_any_failure_cleans_up_and_a_rerun_finishes_exactly_once(
    n_files: int, batch_size: int, prefetch: int, faults: dict[str, int]
) -> None:
    keys = [f"pages/p{i:02d}.jp2" for i in range(n_files)]
    fn = MapFunction(fail_on=faults.get("map"))
    plan = Bag.from_bucket(SRC).map_batches(fn, batch_size=batch_size).to_bucket(OUT)

    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as scratch:
        bucket = FakeBucket().install(mp)
        bucket.seed(SRC, keys, b"page")
        inject(bucket, faults, keys)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the store warns about missing files, as the Hub does
            try:
                plan.compute(prefetch=prefetch, dir=scratch)
            except (Boom, FaultInjected, FileNotFoundError):
                pass
        assert count_bb_dirs(Path(scratch)) == 0
        done_after_first_run = len(bucket.keys(OUT))

        heal(bucket, fn)
        stats = plan.compute(prefetch=prefetch, dir=scratch)

        assert count_bb_dirs(Path(scratch)) == 0
        assert bucket.keys(OUT) == [key + ".md" for key in keys]
        assert (stats.skipped, stats.processed) == (
            done_after_first_run,
            n_files - done_after_first_run,
        )
        written = bucket.uploaded_keys(OUT)
        assert len(written) == len(set(written)), "an output was written twice"
        assert all(bucket.get(OUT, key + ".md") == b"PAGE" for key in keys)
