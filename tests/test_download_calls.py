"""How ``batched_files`` groups batches into hub calls (CONTRACT §2).

Every hub call has a fixed cost, so consecutive batches of sized keys share a call until it
holds about 64 MB or 256 files. A batch is never split; one at or above that size is a call of
its own. String keys carry no size, so each batch is one call. The conftest fixture sets one
file per call (so each batch is a call); these tests set the grouping they check.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from conftest import count_bb_dirs
from fake_bucket import FakeBucket

from bucketbag import LoadedItem, batched_files

pytestmark = pytest.mark.internals  # sets the private call-size constants

SRC = "ns/src"
KEYS = [f"pages/p{i}.jp2" for i in range(9)]


def call_size(monkeypatch: pytest.MonkeyPatch, files: int = 256, nbytes: int = 64_000_000) -> None:
    monkeypatch.setattr("bucketbag._hub._CALL_FILES", files)
    monkeypatch.setattr("bucketbag._hub._CALL_BYTES", nbytes)


def keys_of(batches: Iterable[list[LoadedItem]]) -> list[list[str]]:
    return [[item.key for item in batch] for batch in batches]


def test_small_batches_share_one_call(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch)
    files = bucket.seed(SRC, KEYS)
    got = [
        [(item.key, item.bytes) for item in batch]
        for batch in batched_files(SRC, keys=files, n=1, dir=tmp_path)
    ]
    assert got == [[(key, b"xxxx")] for key in KEYS]
    assert bucket.calls["download"] == 1
    assert count_bb_dirs(tmp_path) == 0


def test_a_call_closes_once_it_reaches_the_file_count(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch, files=3)
    files = bucket.seed(SRC, KEYS)
    got = keys_of(batched_files(SRC, keys=files, n=2, prefetch=1, dir=tmp_path))
    assert got == [KEYS[i : i + 2] for i in range(0, 9, 2)]
    assert bucket.calls["download"] == 3  # batches 1+2, 3+4, then 5 alone
    assert count_bb_dirs(tmp_path) == 0


def test_a_call_closes_once_it_reaches_the_byte_size(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch, nbytes=10)
    files = [bucket.put(SRC, f"big/b{i}.bin", data=b"y" * 6) for i in range(4)]
    assert len(list(batched_files(SRC, keys=files, n=1, dir=tmp_path))) == 4
    assert bucket.calls["download"] == 2  # 6 + 6 bytes reach 10, twice


def test_a_large_batch_is_one_call_and_never_split(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch, nbytes=10)
    files = [bucket.put(SRC, f"big/b{i}.bin", data=b"y" * 25) for i in range(4)]
    assert [len(batch) for batch in batched_files(SRC, keys=files, n=4, dir=tmp_path)] == [4]
    assert bucket.calls["download"] == 1


def test_string_keys_download_one_call_per_batch(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_size(monkeypatch)
    bucket.seed(SRC, KEYS)
    assert [len(batch) for batch in batched_files(SRC, keys=KEYS, n=4, dir=tmp_path)] == [4, 4, 1]
    assert bucket.calls["download"] == 3


@pytest.mark.parametrize("prefetch", [0, 2])
def test_stopping_inside_a_shared_call_deletes_everything(
    bucket: FakeBucket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefetch: int
) -> None:
    call_size(monkeypatch, files=3)
    files = bucket.seed(SRC, KEYS)
    batches = batched_files(SRC, keys=files, n=2, prefetch=prefetch, dir=tmp_path)
    next(batches)  # the first call holds two batches; the second is still unused
    batches.close()
    assert count_bb_dirs(tmp_path) == 0
