"""How ``batched_files`` groups files into hub calls (CONTRACT §2).

Every hub call has a fixed cost, so sized keys are packed about 64 MB (at most 256 files) to a
call, independent of the batch size; a file larger than that gets a call of its own. String
keys carry no size, so each batch is one call. The conftest fixture sets one file per call;
these tests set the grouping they check.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import count_bb_dirs
from fake_bucket import FakeBucket

from bucketbag import batched_files

pytestmark = pytest.mark.internals  # sets the private call-size constants

SRC = "ns/src"
KEYS = [f"pages/p{i}.jp2" for i in range(9)]


def call_size(monkeypatch: pytest.MonkeyPatch, files: int = 256, nbytes: int = 64_000_000) -> None:
    monkeypatch.setattr("bucketbag._hub._CALL_FILES", files)
    monkeypatch.setattr("bucketbag._hub._CALL_BYTES", nbytes)


def keys_and_bytes(batches: Iterator[list]) -> list[list[tuple[str, bytes]]]:
    return [[(item.key, item.bytes) for item in batch] for batch in batches]


def test_small_files_share_one_call_whatever_the_batch_size(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch)
    files = bucket.seed(SRC, KEYS)
    got = keys_and_bytes(batched_files(SRC, keys=files, n=1, dir=tmp_path))
    assert got == [[(key, b"xxxx")] for key in KEYS]
    assert bucket.calls["download"] == 1
    assert count_bb_dirs(tmp_path) == 0


def test_batches_are_cut_across_calls_in_key_order(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch, files=3)
    files = bucket.seed(SRC, KEYS)
    got = keys_and_bytes(batched_files(SRC, keys=files, n=2, prefetch=1, dir=tmp_path))
    assert [[key for key, _ in batch] for batch in got] == [KEYS[i : i + 2] for i in range(0, 9, 2)]
    assert bucket.calls["download"] == 3
    assert count_bb_dirs(tmp_path) == 0


def test_a_file_larger_than_a_call_gets_a_call_of_its_own(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch, nbytes=10)
    files = [bucket.put(SRC, f"big/b{i}.bin", data=b"y" * 25) for i in range(4)]
    batches = list(batched_files(SRC, keys=files, n=4, dir=tmp_path))
    assert [len(batch) for batch in batches] == [4]
    assert bucket.calls["download"] == 4


def test_string_keys_download_one_call_per_batch(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch)
    bucket.seed(SRC, KEYS)
    batches = list(batched_files(SRC, keys=KEYS, n=4, dir=tmp_path))
    assert [len(batch) for batch in batches] == [4, 4, 1]
    assert bucket.calls["download"] == 3


def test_a_batch_spanning_calls_keeps_the_disk_bound(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch, files=2)
    files = bucket.seed(SRC, KEYS)
    for _ in batched_files(SRC, keys=files, n=3, prefetch=1, dir=tmp_path):
        pass
    # prefetch=1 allows 2 calls on disk; a batch of 3 needs 2 calls of 2 files, which fits.
    assert max(bucket.dirs_at_download) <= 2
    assert count_bb_dirs(tmp_path) == 0


@pytest.mark.parametrize("prefetch", [0, 2])
def test_stopping_mid_call_deletes_every_call(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefetch: int
) -> None:
    call_size(monkeypatch, files=3)
    files = bucket.seed(SRC, KEYS)
    batches = batched_files(SRC, keys=files, n=2, prefetch=prefetch, dir=tmp_path)
    next(batches)  # the first call (3 files) is held, one file still unused
    batches.close()
    assert count_bb_dirs(tmp_path) == 0
