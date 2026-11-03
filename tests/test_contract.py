"""The contract in CONTRACT.md, one test per rule, through public entry points only.

Tests marked ``known_gap`` check a rule that today's code breaks. Each fails with ``KnownGap``
(strict xfail) until its fix lands; the fix removes the marker.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import count_bb_dirs, gap, known_gap
from fake_bucket import FakeBucket, FaultInjected

import bucketbag
from bucketbag import Bag, LoadedItem, batched_files, completed_keys, iter_keys, put_files

SRC, OUT = "ns/src", "ns/out"
PAGES = [f"pages/p{i}.jp2" for i in range(6)]

Output = tuple[str, bytes]


def upper_md(items: list[LoadedItem]) -> list[Output]:
    """One output per input, keyed by the input key + ".md" (the resume convention)."""
    return [(it.key + ".md", it.bytes.upper()) for it in items]


def bag(batch_size: int = 2) -> Bag:
    return Bag.from_bucket(SRC).map_batches(upper_md, batch_size=batch_size).to_bucket(OUT)


def out_keys(bucket: FakeBucket) -> list[str]:
    return bucket.keys(OUT)


# --- 1. Addressing ------------------------------------------------------------------------


@pytest.fixture
def tree(bucket: FakeBucket) -> FakeBucket:
    bucket.seed("ns/b", ["a/b/x.jp2", "a/b/y.jp2", "a/bc/z.jp2", "c.jp2"])
    return bucket


def test_trailing_slash_prefix_is_one_directory(tree: FakeBucket) -> None:
    assert list(iter_keys("ns/b", prefix="a/b/")) == ["a/b/x.jp2", "a/b/y.jp2"]


def test_prefix_without_slash_also_matches_siblings(tree: FakeBucket) -> None:
    assert list(iter_keys("ns/b", prefix="a/b")) == ["a/b/x.jp2", "a/b/y.jp2", "a/bc/z.jp2"]


@known_gap("_parse_bucket drops the trailing slash of a prefix inside the bucket reference")
def test_trailing_slash_inside_bucket_reference_is_one_directory(tree: FakeBucket) -> None:
    keys = list(iter_keys("hf://buckets/ns/b/a/b/"))
    if keys != ["a/b/x.jp2", "a/b/y.jp2"]:
        gap(f"'hf://buckets/ns/b/a/b/' listed {keys}")


@known_gap("a Hub web URL is parsed as bucket 'https:/'")
def test_web_url_is_a_bucket_reference(tree: FakeBucket) -> None:
    keys = list(iter_keys("https://huggingface.co/buckets/ns/b/tree/a/b/"))
    if keys != ["a/b/x.jp2", "a/b/y.jp2"]:
        gap(f"the web URL listed {keys}")


def test_globs_match_the_whole_key_from_the_bucket_root(tree: FakeBucket) -> None:
    # `*.jp2` matches root-level keys only: `*` does not cross `/`, and the glob is not
    # relative to the prefix. Callers build `include=f"{prefix}{pattern}"` on this rule.
    assert list(iter_keys("ns/b", include="*.jp2")) == ["c.jp2"]
    assert list(iter_keys("ns/b", prefix="a/b/", include="*.jp2")) == []
    assert list(iter_keys("ns/b", prefix="a/b/", include="a/b/*.jp2")) == ["a/b/x.jp2", "a/b/y.jp2"]
    assert len(list(iter_keys("ns/b", include="**/*.jp2"))) == 4


def test_exclude_drops_matching_keys(tree: FakeBucket) -> None:
    keys = list(iter_keys("ns/b", include="**/*.jp2", exclude="a/bc/**"))
    assert keys == ["a/b/x.jp2", "a/b/y.jp2", "c.jp2"]


def test_listings_return_full_keys_sorted(tree: FakeBucket) -> None:
    tree.listing_order = "reversed"
    assert list(iter_keys("ns/b", prefix="a/")) == ["a/b/x.jp2", "a/b/y.jp2", "a/bc/z.jp2"]


def test_put_files_writes_under_the_prefix_in_the_reference(bucket: FakeBucket) -> None:
    put_files([("k.md", "text")], "hf://buckets/ns/out/run1")
    assert bucket.keys(OUT) == ["run1/k.md"]


def test_start_after_and_limit_select_a_slice_of_the_sorted_keys(tree: FakeBucket) -> None:
    assert list(iter_keys("ns/b", start_after="a/b/x.jp2", limit=2)) == ["a/b/y.jp2", "a/bc/z.jp2"]


def test_folder_entries_are_skipped(tree: FakeBucket) -> None:
    tree.folders = True
    assert list(iter_keys("ns/b", prefix="a/")) == ["a/b/x.jp2", "a/b/y.jp2", "a/bc/z.jp2"]


# --- 2. Temporary files: tests/test_batched_files.py covers each prefetch level -----------


def test_a_kept_generator_deletes_its_files_on_close(bucket: FakeBucket, tmp_path: Path) -> None:
    files = bucket.seed(SRC, PAGES)
    batches = batched_files(SRC, keys=files, n=1, prefetch=2, dir=tmp_path)
    next(batches)
    assert count_bb_dirs(tmp_path) > 0  # the current batch and the lookahead are on disk
    batches.close()
    assert count_bb_dirs(tmp_path) == 0


# --- 3. Resume ----------------------------------------------------------------------------


def test_rerun_of_a_finished_run_processes_nothing(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES, b"page")
    first = bag().compute(prefetch=0, dir=tmp_path)
    second = bag().compute(prefetch=0, dir=tmp_path)
    assert (first.processed, second.processed, second.skipped) == (6, 0, 6)
    assert out_keys(bucket) == [key + ".md" for key in PAGES]
    assert bucket.get(OUT, "pages/p0.jp2.md") == b"PAGE"


def test_output_under_a_run_prefix_marks_its_input_done(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES, b"page")
    run = Bag.from_bucket(SRC).map_batches(upper_md).to_bucket(f"{OUT}/run1")
    run.compute(prefetch=0, dir=tmp_path)
    assert run.compute(prefetch=0, dir=tmp_path).processed == 0


def test_output_key_without_the_input_key_is_reported_once(
    bucket: FakeBucket, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def replace_extension(items: list[LoadedItem]) -> list[Output]:
        return [(it.key.removesuffix(".jp2") + ".md", b"text") for it in items]

    bucket.seed(SRC, PAGES[:4])
    run = Bag.from_bucket(SRC).map_batches(replace_extension, batch_size=2).to_bucket(OUT)
    with caplog.at_level(logging.WARNING, logger="bucketbag"):
        run.compute(prefetch=0, dir=tmp_path)
    warnings = [record for record in caplog.records if "resume" in record.getMessage()]
    assert len(warnings) == 1  # two batches, one warning
    assert "pages/p0.md" in warnings[0].getMessage()


# --- 4. Failures --------------------------------------------------------------------------


class Boom(RuntimeError):
    pass


def fail_on_batch(number: int) -> Callable[[list[LoadedItem]], list[Output]]:
    """A map function that raises on its ``number``-th call (1-based)."""
    calls = 0

    def fn(items: list[LoadedItem]) -> list[Output]:
        nonlocal calls
        calls += 1
        if calls == number:
            raise Boom(f"batch {number}")
        return upper_md(items)

    return fn


def run_expecting(error: type[BaseException], run: Bag, tmp_path: Path, prefetch: int = 2) -> None:
    with pytest.raises(error):
        run.compute(prefetch=prefetch, dir=tmp_path)
    assert count_bb_dirs(tmp_path) == 0


def assert_rerun_finishes_the_rest(bucket: FakeBucket, tmp_path: Path, done_before: int) -> None:
    stats = bag().compute(prefetch=2, dir=tmp_path)
    assert (stats.skipped, stats.processed) == (done_before, len(PAGES) - done_before)
    assert out_keys(bucket) == [key + ".md" for key in PAGES]
    written = bucket.uploaded_keys(OUT)
    assert len(written) == len(set(written)), "an output was written twice"


def test_map_function_error_raises_and_keeps_earlier_outputs(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    bucket.seed(SRC, PAGES)
    failing = Bag.from_bucket(SRC).map_batches(fail_on_batch(2), batch_size=2).to_bucket(OUT)
    run_expecting(Boom, failing, tmp_path)
    assert out_keys(bucket) == ["pages/p0.jp2.md", "pages/p1.jp2.md"]
    assert_rerun_finishes_the_rest(bucket, tmp_path, done_before=2)


def test_download_error_raises_and_keeps_earlier_outputs(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    bucket.seed(SRC, PAGES)
    bucket.fail_download_on.add(3)  # without prefetch, download call 3 is batch 3
    run_expecting(FaultInjected, bag(), tmp_path, prefetch=0)
    bucket.fail_download_on.clear()
    assert len(out_keys(bucket)) == 4
    assert_rerun_finishes_the_rest(bucket, tmp_path, done_before=4)


def test_upload_error_raises_and_that_batch_writes_nothing(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    bucket.seed(SRC, PAGES)
    bucket.fail_upload_on.add(2)
    run_expecting(FaultInjected, bag(), tmp_path)
    bucket.fail_upload_on.clear()
    assert out_keys(bucket) == ["pages/p0.jp2.md", "pages/p1.jp2.md"]
    assert_rerun_finishes_the_rest(bucket, tmp_path, done_before=2)


def test_partial_upload_raises_and_what_landed_counts_as_done(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    bucket.seed(SRC, PAGES)
    bucket.partial_upload_on.add(2)  # batch 2 stores 1 of its 2 outputs, then fails
    run_expecting(FaultInjected, bag(), tmp_path)
    bucket.partial_upload_on.clear()
    assert out_keys(bucket) == ["pages/p0.jp2.md", "pages/p1.jp2.md", "pages/p2.jp2.md"]
    assert_rerun_finishes_the_rest(bucket, tmp_path, done_before=3)


def test_string_key_gone_at_download_is_dropped_from_its_batch(
    bucket: FakeBucket, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The hub checks string keys and skips missing ones with a warning. A BucketFile (what Bag
    # passes) skips that check and downloads by content hash, so this rule is for string keys.
    bucket.seed(SRC, PAGES)
    bucket.missing.add("pages/p3.jp2")
    with pytest.warns(UserWarning, match="p3.jp2"), caplog.at_level(logging.WARNING, "bucketbag"):
        batches = list(batched_files(SRC, keys=list(PAGES), n=10, prefetch=0, dir=tmp_path))
    assert [[it.key for it in batch] for batch in batches] == [PAGES[:3] + PAGES[4:]]
    assert any("missing" in record.getMessage() for record in caplog.records)


# --- 5. Arguments -------------------------------------------------------------------------


def test_bag_bounds_each_batch_by_bytes(bucket: FakeBucket, tmp_path: Path) -> None:
    for i, size in enumerate([3, 3, 3, 9, 3]):
        bucket.put(SRC, f"pages/p{i}.jp2", b"x" * size)
    seen: list[list[str]] = []

    def record(items: list[LoadedItem]) -> list[Output]:
        seen.append([it.key.removeprefix("pages/") for it in items])
        return upper_md(items)

    run = Bag.from_bucket(SRC).map_batches(record, batch_size=None, max_bytes=6).to_bucket(OUT)
    run.compute(prefetch=0, dir=tmp_path)
    # 3+3 fits in 6 bytes; the 9-byte file is larger than the bound, so it is a batch alone.
    assert seen == [["p0.jp2", "p1.jp2"], ["p2.jp2"], ["p3.jp2"], ["p4.jp2"]]


def test_token_reaches_every_hub_call(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES[:2])
    run = Bag.from_bucket(SRC, token="hf_test").map_batches(upper_md).to_bucket(OUT)
    run.compute(prefetch=0, dir=tmp_path)
    assert {name for name, _ in bucket.tokens} == {
        "list_bucket_tree",
        "download_bucket_files",
        "batch_bucket_files",
    }
    assert {token for _, token in bucket.tokens} == {"hf_test"}


@pytest.mark.parametrize(("n", "max_bytes"), [(None, None), (0, None), (None, 0)])
def test_unworkable_batch_bounds_are_refused_before_listing(
    bucket: FakeBucket, tmp_path: Path, n: int | None, max_bytes: int | None
) -> None:
    bucket.seed(SRC, PAGES)
    with pytest.raises(ValueError):
        next(batched_files(SRC, n=n, max_bytes=max_bytes, dir=tmp_path))
    with pytest.raises(ValueError):
        Bag.from_bucket(SRC).map_batches(upper_md, batch_size=n, max_bytes=max_bytes)
    assert bucket.calls["list"] == 0


def test_negative_prefetch_is_refused(bucket: FakeBucket, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prefetch"):
        next(batched_files(SRC, prefetch=-1, dir=tmp_path))


def test_map_function_with_keyword_only_options(bucket: FakeBucket, tmp_path: Path) -> None:
    def fn(items: list[LoadedItem], *, suffix: str = ".md") -> Iterable[Output]:
        return [(it.key + suffix, b"x") for it in items]

    bucket.seed(SRC, PAGES[:1])
    Bag.from_bucket(SRC).map_batches(fn).to_bucket(OUT).compute(prefetch=0, dir=tmp_path)
    assert bucket.keys(OUT) == ["pages/p0.jp2.md"]


def test_completed_keys_refuses_a_shard_without_the_column(
    bucket: FakeBucket, fake_parquet: SimpleNamespace
) -> None:
    bucket.seed(OUT, ["run/part-0.parquet"])
    fake_parquet.columns = ["source_key"]
    with pytest.raises(ValueError, match="__source_key"):
        completed_keys(OUT)


def test_completed_keys_skips_an_unreadable_shard(
    bucket: FakeBucket, fake_parquet: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    bucket.seed(OUT, ["run/part-0.parquet", "run/part-1.parquet"])
    fake_parquet.unreadable.add("run/part-1.parquet")
    with caplog.at_level(logging.WARNING, logger="bucketbag"):
        assert completed_keys(OUT) == {"run/part-0.parquet"}
    assert any("part-1" in record.getMessage() for record in caplog.records)


# --- Public names -------------------------------------------------------------------------


def test_public_names() -> None:
    # A module split must keep every name importable. Removing one is an API change: edit here.
    assert sorted(bucketbag.__all__) == [
        "Bag",
        "BagStats",
        "LoadedItem",
        "batched_files",
        "boost",
        "completed_keys",
        "iter_keys",
        "partition_all",
        "put_bytes",
        "put_files",
        "put_text",
    ]
    assert all(hasattr(bucketbag, name) for name in bucketbag.__all__)
