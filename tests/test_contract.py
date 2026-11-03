"""The contract in CONTRACT.md, one test per rule, through public entry points only.

A rule that the code breaks gets a test marked ``known_gap`` (see conftest.py): a strict xfail
that only ``KnownGap`` satisfies, so the fix has to remove the marker. None is open today.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import count_bb_dirs
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
    return (
        Bag.from_bucket(SRC).map_batches(upper_md, batch_size=batch_size).to_bucket(OUT, done=".md")
    )


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


def test_trailing_slash_inside_bucket_reference_is_one_directory(tree: FakeBucket) -> None:
    assert list(iter_keys("hf://buckets/ns/b/a/b/")) == ["a/b/x.jp2", "a/b/y.jp2"]
    assert list(iter_keys("ns/b/a/b/")) == ["a/b/x.jp2", "a/b/y.jp2"]


def test_web_url_is_a_bucket_reference(tree: FakeBucket) -> None:
    keys = list(iter_keys("https://huggingface.co/buckets/ns/b/tree/a/b/"))
    assert keys == ["a/b/x.jp2", "a/b/y.jp2"]


def test_prefix_joins_below_the_prefix_in_the_reference(tree: FakeBucket) -> None:
    # "ns/b/a" with prefix="b/" lists a/b/ only: the two prefixes join as directories.
    assert list(iter_keys("ns/b/a", prefix="b/")) == ["a/b/x.jp2", "a/b/y.jp2"]
    assert list(iter_keys("ns/b/a/", prefix="")) == ["a/b/x.jp2", "a/b/y.jp2", "a/bc/z.jp2"]


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


@pytest.mark.parametrize("prefetch", [0, 1, 2, 3])
def test_at_most_prefetch_plus_one_batches_are_on_disk(
    bucket: FakeBucket, tmp_path: Path, prefetch: int
) -> None:
    files = bucket.seed(SRC, [f"pages/p{i:02d}.jp2" for i in range(10)])
    for _ in batched_files(SRC, keys=files, n=1, prefetch=prefetch, dir=tmp_path):
        time.sleep(0.02)  # hold the batch, so downloads ahead start while it is on disk
    # The fake counts batch dirs as each download starts, so a peak between yields counts too.
    assert max(bucket.dirs_at_download) <= prefetch + 1
    if prefetch:
        assert max(bucket.dirs_at_download) >= 2  # the lookahead does download ahead


def test_a_worker_error_does_not_stop_the_cleanup(bucket: FakeBucket, tmp_path: Path) -> None:
    files = bucket.seed(SRC, PAGES)
    bucket.fail_download_keys.add(PAGES[1])  # batch 2: queued behind batch 1, ahead of 3 and 4
    bucket.download_error = SystemExit  # a BaseException stored in a queued download
    batches = batched_files(SRC, keys=files, n=1, prefetch=3, dir=tmp_path)
    next(batches)
    deadline = time.monotonic() + 5
    while bucket.calls["download"] < 4 and time.monotonic() < deadline:
        time.sleep(0.005)  # until all four queued downloads, batch 2 included, have started
    assert bucket.calls["download"] == 4
    batches.close()
    assert count_bb_dirs(tmp_path) == 0


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs SIGALRM")
@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_ctrl_c_while_waiting_leaves_no_files(
    bucket: FakeBucket, tmp_path: Path, prefetch: int
) -> None:
    files = bucket.seed(SRC, PAGES)
    bucket.download_seconds = 0.3
    batches = batched_files(SRC, keys=files, n=1, prefetch=prefetch, dir=tmp_path)

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.05)  # fires while next() waits for the download
    try:
        with pytest.raises(KeyboardInterrupt):
            next(batches)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert count_bb_dirs(tmp_path) == 0


def test_a_kept_generator_deletes_its_files_on_close(bucket: FakeBucket, tmp_path: Path) -> None:
    files = bucket.seed(SRC, PAGES)
    batches = batched_files(SRC, keys=files, n=1, prefetch=2, dir=tmp_path)
    next(batches)
    assert count_bb_dirs(tmp_path) > 0  # the current batch and the lookahead are on disk
    batches.close()
    assert count_bb_dirs(tmp_path) == 0


# --- 3. Resume ----------------------------------------------------------------------------


def test_resume_needs_a_done_suffix(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES[:1])
    run = Bag.from_bucket(SRC).map_batches(upper_md).to_bucket(OUT)
    with pytest.raises(ValueError, match="done="):
        run.compute(prefetch=0, dir=tmp_path)
    assert bucket.calls["list"] == 0
    assert run.compute(resume=False, prefetch=0, dir=tmp_path).processed == 1


def test_an_input_is_done_only_by_its_own_output(bucket: FakeBucket, tmp_path: Path) -> None:
    # pages/p10.md must not mark pages/p1 done, even when limit= hides pages/p10 from the plan.
    bucket.seed(SRC, ["pages/p1", "pages/p10"])
    bucket.seed(OUT, ["pages/p10.md"])
    for plan in (Bag.from_bucket(SRC), Bag.from_bucket(SRC, limit=1)):
        stats = (
            plan.map_batches(upper_md)
            .to_bucket(OUT, done=".md")
            .compute(resume=True, prefetch=0, dir=tmp_path)
        )
        assert stats.processed == 1
        bucket.store[OUT].pop("pages/p1.md")  # undo the write for the next plan


def test_rerun_of_a_finished_run_processes_nothing(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES, b"page")
    first = bag().compute(prefetch=0, dir=tmp_path)
    second = bag().compute(prefetch=0, dir=tmp_path)
    assert (first.processed, second.processed, second.skipped) == (6, 0, 6)
    assert out_keys(bucket) == [key + ".md" for key in PAGES]
    assert bucket.get(OUT, "pages/p0.jp2.md") == b"PAGE"


def test_output_under_a_run_prefix_marks_its_input_done(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES, b"page")
    run = Bag.from_bucket(SRC).map_batches(upper_md).to_bucket(f"{OUT}/run1", done=".md")
    run.compute(prefetch=0, dir=tmp_path)
    assert run.compute(prefetch=0, dir=tmp_path).processed == 0


@pytest.mark.parametrize("entry", ["take", "compute"])
def test_an_input_without_its_done_output_is_reported_once(
    bucket: FakeBucket, tmp_path: Path, caplog: pytest.LogCaptureFixture, entry: str
) -> None:
    def replace_extension(items: list[LoadedItem]) -> list[Output]:
        return [(it.key.removesuffix(".jp2") + ".md", b"text") for it in items]

    bucket.seed(SRC, PAGES[:4])
    run = (
        Bag.from_bucket(SRC).map_batches(replace_extension, batch_size=2).to_bucket(OUT, done=".md")
    )
    with caplog.at_level(logging.WARNING, logger="bucketbag"):
        if entry == "take":
            run.take(4, dir=tmp_path)
        else:
            run.compute(prefetch=0, dir=tmp_path)
    warnings = [record for record in caplog.records if "resume" in record.getMessage()]
    assert len(warnings) == 1  # two batches, one warning
    assert "pages/p0.jp2" in warnings[0].getMessage()


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


@pytest.mark.parametrize("entry", ["take", "compute"])
def test_a_kept_exception_does_not_keep_temp_files(
    bucket: FakeBucket, tmp_path: Path, entry: str
) -> None:
    bucket.seed(SRC, PAGES)
    failing = (
        Bag.from_bucket(SRC).map_batches(fail_on_batch(2), batch_size=2).to_bucket(OUT, done=".md")
    )
    kept: list[BaseException] = []  # a notebook or a retry loop keeps the error and its traceback
    try:
        if entry == "take":
            failing.take(6, dir=tmp_path)
        else:
            failing.compute(prefetch=2, dir=tmp_path)
    except Boom as error:
        kept.append(error)
    assert kept
    assert count_bb_dirs(tmp_path) == 0


def test_map_function_error_raises_and_keeps_earlier_outputs(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    bucket.seed(SRC, PAGES)
    failing = (
        Bag.from_bucket(SRC).map_batches(fail_on_batch(2), batch_size=2).to_bucket(OUT, done=".md")
    )
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

    run = (
        Bag.from_bucket(SRC)
        .map_batches(record, batch_size=None, max_bytes=6)
        .to_bucket(OUT, done=".md")
    )
    # 3+3 fits in 6 bytes; the 9-byte file is larger than the bound, so it is a batch alone.
    expected = [["p0.jp2", "p1.jp2"], ["p2.jp2"], ["p3.jp2"], ["p4.jp2"]]
    run.take(5, dir=tmp_path)
    assert seen == expected
    seen.clear()
    run.compute(prefetch=0, dir=tmp_path)
    assert seen == expected


def test_token_reaches_every_hub_call(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES[:2])
    run = Bag.from_bucket(SRC, token="hf_test").map_batches(upper_md).to_bucket(OUT, done=".md")
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


@pytest.mark.parametrize(
    ("n", "max_bytes", "prefetch"),
    [
        (True, None, 2),
        (2.5, None, 2),
        (None, float("nan"), 2),
        (None, float("inf"), 2),
        (None, True, 2),
        (4, None, 1.5),
        (4, None, None),
    ],
)
def test_batch_settings_of_the_wrong_kind_are_refused(
    bucket: FakeBucket, tmp_path: Path, n: Any, max_bytes: Any, prefetch: Any
) -> None:
    with pytest.raises(TypeError, match="whole number|finite number"):
        next(batched_files(SRC, n=n, max_bytes=max_bytes, prefetch=prefetch, dir=tmp_path))
    assert bucket.calls["list"] == 0


def test_a_float_byte_cap_works(bucket: FakeBucket, tmp_path: Path) -> None:
    files = bucket.seed(SRC, PAGES)  # 4 bytes each
    batches = list(batched_files(SRC, keys=files, n=None, max_bytes=8e0, prefetch=0, dir=tmp_path))
    assert [len(batch) for batch in batches] == [2, 2, 2]


def test_negative_prefetch_is_refused_before_listing_or_setup(
    bucket: FakeBucket, tmp_path: Path
) -> None:
    bucket.seed(SRC, PAGES)
    setups: list[int] = []

    def load_model() -> None:
        setups.append(1)

    def fn(items: list[LoadedItem], ctx: None) -> list[Output]:
        return upper_md(items)

    run = Bag.from_bucket(SRC).map_batches(fn, setup=load_model).to_bucket(OUT, done=".md")
    with pytest.raises(ValueError, match="prefetch"):
        next(batched_files(SRC, prefetch=-1, dir=tmp_path))
    with pytest.raises(ValueError, match="prefetch"):
        run.compute(prefetch=-1, dir=tmp_path)
    assert (bucket.calls["list"], setups) == (0, [])


def test_map_function_with_keyword_only_options(bucket: FakeBucket, tmp_path: Path) -> None:
    def fn(items: list[LoadedItem], *, suffix: str = ".md") -> Iterable[Output]:
        return [(it.key + suffix, b"x") for it in items]

    bucket.seed(SRC, PAGES[:1])
    Bag.from_bucket(SRC).map_batches(fn).to_bucket(OUT, done=".md").compute(
        prefetch=0, dir=tmp_path
    )
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


# --- 6. Run report ----------------------------------------------------------------------


SLOW = 0.1  # per batch in the slow stage; the fake's other stages take well under 1 ms


def slow_map(items: list[LoadedItem]) -> list[Output]:
    time.sleep(SLOW)
    return upper_md(items)


@pytest.mark.parametrize(
    ("stage", "words"),
    [
        ("download", "prefetch"),
        ("map", "map function"),
        ("upload", "Uploads"),
    ],
)
def test_stats_name_the_slowest_stage_and_what_to_change(
    bucket: FakeBucket, tmp_path: Path, stage: str, words: str
) -> None:
    bucket.seed(SRC, PAGES)
    fn = upper_md
    if stage == "download":
        bucket.download_seconds = SLOW
    elif stage == "map":
        fn = slow_map
    else:
        bucket.upload_seconds = SLOW
    run = Bag.from_bucket(SRC).map_batches(fn, batch_size=2).to_bucket(OUT, done=".md")
    stats = run.compute(prefetch=0, dir=tmp_path)
    assert stats.bound_by == stage
    assert words in stats.hint
    assert f"bound by {stage}" in str(stats)


def test_seconds_covers_listing_and_setup(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES[:1])
    bucket.list_seconds = SLOW  # the source and the output listing

    def fn(items: list[LoadedItem], ctx: None) -> list[Output]:
        return upper_md(items)

    run = Bag.from_bucket(SRC).map_batches(fn, setup=lambda: time.sleep(SLOW))
    stats = run.to_bucket(OUT, done=".md").compute(prefetch=0, dir=tmp_path)
    assert stats.seconds >= 3 * SLOW
    assert stats.wait_s + stats.map_s + stats.upload_s < SLOW  # the split leaves both out


def test_a_run_with_nothing_to_do_skips_setup(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES)
    bag().compute(prefetch=0, dir=tmp_path)
    setups: list[int] = []

    def fn(items: list[LoadedItem], ctx: None) -> list[Output]:
        return upper_md(items)

    run = Bag.from_bucket(SRC).map_batches(fn, setup=lambda: setups.append(1))
    stats = run.to_bucket(OUT, done=".md").compute(prefetch=0, dir=tmp_path)
    assert (stats.skipped, stats.processed, setups) == (len(PAGES), 0, [])


def test_a_run_with_nothing_to_do_has_no_verdict(bucket: FakeBucket, tmp_path: Path) -> None:
    bucket.seed(SRC, PAGES)
    bag().compute(prefetch=0, dir=tmp_path)
    stats = bag().compute(prefetch=0, dir=tmp_path)  # everything is done
    assert (stats.batches, stats.bound_by, stats.hint) == (0, None, "")
    assert "bound by" not in str(stats)


def test_progress_is_logged_with_the_time_split(
    bucket: FakeBucket,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bucketbag.bag, "_PROGRESS_EVERY_S", 0.0)  # log after every batch
    bucket.seed(SRC, PAGES)
    with caplog.at_level(logging.INFO, logger="bucketbag"):
        bag().compute(prefetch=0, dir=tmp_path)
    lines = [r.getMessage() for r in caplog.records if "inputs" in r.getMessage()]
    assert lines[-1].startswith("6/6 inputs")
    assert "download wait" in lines[-1] and "map" in lines[-1]
    messages = [r.getMessage() for r in caplog.records]
    assert any("batch_size=2" in m and "prefetch=0" in m for m in messages)  # settings at start
    assert "bound by" in messages[-1]  # the run report at the end


def test_the_progress_line_states_rate_and_time_left(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="bucketbag"):
        bucketbag.bag._log_progress(50, 150, 100.0, "download wait 5%, map 90%, upload 5%")
        bucketbag.bag._log_progress(0, 150, 0.0, "download wait 0%, map 0%, upload 0%")
    first, second = (r.getMessage() for r in caplog.records)
    assert (
        first
        == "50/150 inputs, 0.5/s, about 3 min left; time: download wait 5%, map 90%, upload 5%"
    )
    assert second.startswith("0/150 inputs, 0.0/s; time:")  # no rate yet, so no time left


def test_logs_reach_stderr_until_the_caller_configures_logging(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    log = logging.getLogger("bucketbag")
    monkeypatch.setattr(logging.root, "handlers", [])  # as in a script with no logging setup
    log.info("hello")
    assert "[bucketbag] hello" in capsys.readouterr().err
    monkeypatch.setattr(logging.root, "handlers", [logging.NullHandler()])  # the caller's setup
    log.info("hello again")
    assert "hello again" not in capsys.readouterr().err


def test_a_too_small_scratch_dir_is_reported(
    bucket: FakeBucket,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = bucket.seed(SRC, PAGES)  # 4 bytes each; with n=2 and prefetch=2, up to 24 on disk

    def disk_with(free: int) -> Callable[[object], SimpleNamespace]:
        return lambda path: SimpleNamespace(total=free, used=0, free=free)

    for free, warns in ((23, True), (24, False)):
        monkeypatch.setattr(bucketbag._hub.shutil, "disk_usage", disk_with(free))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="bucketbag"):
            batches = list(batched_files(SRC, keys=files, n=2, prefetch=2, dir=tmp_path))
        assert len(batches) == 3  # the check reads the batches without using them up
        assert any("GB is free" in r.getMessage() for r in caplog.records) is warns


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
