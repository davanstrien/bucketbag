"""The download window's byte budget (``prefetch_bytes``, CONTRACT §2).

Sized keys download ahead while the bytes on disk and on their way stay within the budget,
capped at half the scratch dir's free space. One call always runs when the next batch needs
it. String keys carry no size, so only the call count applies to them. The conftest fixture
gives one file per call unless a test sets the grouping.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import count_bb_dirs
from fake_bucket import FakeBucket

import bucketbag
from bucketbag import batched_files

SRC = "ns/src"
KEYS = [f"pages/p{i:02d}.bin" for i in range(12)]


def seed(bucket: FakeBucket, size: int = 10) -> list:
    return [bucket.put(SRC, key, data=b"z" * size) for key in KEYS]


def drain_slowly(batches) -> int:  # noqa: ANN001 - any batch iterator
    count = 0
    for batch in batches:
        count += len(batch)
        time.sleep(0.01)  # hold each batch, so downloads ahead get the chance to start
    return count


def test_the_budget_bounds_the_calls_on_disk(bucket: FakeBucket, tmp_path: Path) -> None:
    files = seed(bucket)  # 10 bytes each, one per call
    assert drain_slowly(batched_files(SRC, keys=files, n=1, prefetch_bytes=30, dir=tmp_path)) == 12
    assert 2 <= max(bucket.dirs_at_download) <= 3  # 3 calls of 10 bytes fit in 30
    assert count_bb_dirs(tmp_path) == 0


def test_a_call_larger_than_the_budget_downloads_on_its_own(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    files = seed(bucket)
    assert drain_slowly(batched_files(SRC, keys=files, n=1, prefetch_bytes=5, dir=tmp_path)) == 12
    assert max(bucket.dirs_at_download) == 1


def test_a_zero_budget_downloads_one_call_at_a_time(bucket: FakeBucket, tmp_path: Path) -> None:
    files = seed(bucket)
    got = [
        it.key
        for batch in batched_files(SRC, keys=files, n=1, prefetch_bytes=0, dir=tmp_path)
        for it in batch
    ]
    assert got == KEYS
    assert max(bucket.dirs_at_download) == 1


@pytest.mark.internals
def test_the_budget_is_capped_at_half_the_free_space(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = seed(bucket)
    monkeypatch.setattr(
        bucketbag._hub.shutil, "disk_usage", lambda path: SimpleNamespace(total=40, used=0, free=40)
    )
    drain_slowly(batched_files(SRC, keys=files, n=1, dir=tmp_path))  # default budget: 2 GB
    assert max(bucket.dirs_at_download) <= 2  # half of 40 bytes holds 2 calls of 10


@pytest.mark.internals
def test_a_small_budget_makes_smaller_calls(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bucketbag._hub._CALL_FILES", 256)
    files = [bucket.put(SRC, f"tiny/t{i:02d}.bin", data=b"t") for i in range(40)]
    batches = list(batched_files(SRC, keys=files, n=1, prefetch_bytes=40, dir=tmp_path))
    assert len(batches) == 40
    assert bucket.calls["download"] == 4  # calls of budget / 4 = 10 bytes


def test_string_keys_use_the_call_count(bucket: FakeBucket, tmp_path: Path) -> None:
    seed(bucket)
    batches = batched_files(SRC, keys=KEYS, n=1, prefetch_bytes=1, dir=tmp_path)
    assert drain_slowly(batches) == 12
    assert 2 <= max(bucket.dirs_at_download) <= 3  # default prefetch 2 for string keys


def test_prefetch_still_caps_the_calls(bucket: FakeBucket, tmp_path: Path) -> None:
    files = seed(bucket)
    drain_slowly(batched_files(SRC, keys=files, n=1, prefetch=1, dir=tmp_path))
    assert max(bucket.dirs_at_download) <= 2


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, "2e9"])
def test_a_budget_of_the_wrong_kind_is_refused(
    bucket: FakeBucket, tmp_path: Path, bad: Any
) -> None:
    with pytest.raises(TypeError, match="prefetch_bytes"):
        next(batched_files(SRC, n=1, prefetch_bytes=bad, dir=tmp_path))
    assert bucket.calls["list"] == 0


def test_a_negative_budget_is_refused(bucket: FakeBucket, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prefetch_bytes"):
        next(batched_files(SRC, n=1, prefetch_bytes=-1, dir=tmp_path))
