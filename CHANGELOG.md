# Changelog

## 0.4.0 — 2026-11

A rewrite of the internals around a written contract (CONTRACT.md), plus a run report for
agents. First release on PyPI: `pip install bucketbag`.

### Breaking changes and how to migrate

- **`Bag` resume needs a done-suffix.** Write `to_bucket(out, done=".md")` when each input `k`
  writes `k + ".md"`. An input is done only when that exact output exists, so `pages/p10.md`
  no longer marks `pages/p1` done. `compute(resume=True)` without `done=` raises; pass
  `resume=False` to process everything.
- **The map function is called as `fn(items)`, or `fn(items, ctx)` with `setup=`.** bucketbag
  no longer inspects its signature, so a two-parameter `fn` without `setup=` now gets one
  argument (before, it got `None` as the second).
- **`prefix=` joins below a prefix inside the bucket reference**, as a directory:
  `iter_keys("ns/b/out", prefix="part-")` lists `out/part-*` (before, `prefix=` replaced it).
  `prefix=""` adds nothing; before, it listed the whole bucket.
- **A trailing `/` in the bucket reference is kept**: `"ns/b/a/b/"` lists the directory
  `a/b` only (before, it also matched `a/bc/`).
- **Globs follow fsspec's rules** (`HfFileSystem.glob`): `[...]` is a character class, `*` no
  longer matches an empty path segment, and `**` inside a name (`**.jp2`) raises; write
  `**/*.jp2`.
- **Settings that cannot work raise before anything is listed**: neither `n`/`batch_size` nor
  `max_bytes` set, a value below 1, a negative `prefetch`, a bool, nan or inf.
- **`completed_keys` raises when a readable shard lacks `column=`** (before, every read error
  became a warning, so a wrong column silently re-ran everything).
- **`batched_files(max_workers=)` is removed**; `prefetch` sets the number of download threads.
- **`LoadedItem` is frozen.** Assigning to `item.key` or `item.path` raises
  `FrozenInstanceError`; build a new `LoadedItem(key=..., path=...)` instead.
- **Dependency floors:** `huggingface-hub>=1.21`, `fsspec>=2023.12`.

### New

- Hub web URLs work as bucket references (`https://huggingface.co/buckets/ns/b/tree/x/`).
- A run report: `BagStats` splits the time into download wait, map and upload; `bound_by`
  names the slowest stage and `hint` the setting to change. `compute()` logs its settings,
  a progress line after batches (at most every 30 s) and the report. bucketbag's log lines
  print to stderr as `[bucketbag] …` until your script configures logging.
- A warning before the first download when the largest `prefetch + 1` batches would not fit in
  the scratch dir (on `/dev/shm`, running out kills the process).
- `Bag.compute()` warns once when an input wrote no `k + done` output (it would rerun forever).
- `py.typed`: the package is fully typed.

### Fixed

- `batched_files` kept `prefetch + 2` batches on disk; it now keeps at most `prefetch + 1`.
- Temp files no longer leak when Ctrl-C arrives during a download wait, or when a queued
  download stored a `BaseException`.
- `Bag.take()` honours `max_bytes`.
- `completed_keys` reads each shard's footer once.

### Internal

- A pure core (`_core`) inside a thin I/O shell (`_hub`); one owner per temp dir; one run loop
  for `take()` and `compute()`.
- CONTRACT.md states every rule; `tests/test_contract.py` checks each one against an in-memory
  Hub that binds every call to the real `HfApi` signatures, plus Hypothesis fault fuzzing.
- CI: ruff format, ty on src and tests, a complexity limit, and a 15-minute job timeout.

## 0.3.1 — 2026-08-26

- Fix: `prefix` over-matched sibling directories in `iter_keys`, `batched_files` and
  `completed_keys`. `prefix="a/b/"` returned `a/bc/y` too. Two causes: bucketbag stripped the
  trailing slash before calling the Hub, and the Hub's pagination `Link: next` URL carries a
  literal `/` that a 302 removes, so pages after the first of a recursive listing match on the
  slash-less string prefix. The slash is now sent verbatim and keys are re-filtered
  client-side with the caller's original prefix: a trailing `/` means "this directory"; no
  slash keeps string-prefix semantics.

## 0.3.0

- `max_bytes` raises (not warns) when it cannot be honored (string keys carry no size).
- See git tags `v0.3.0` / `v0.2.0` for earlier release notes.
