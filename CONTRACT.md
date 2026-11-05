# bucketbag contract

What bucketbag promises. `tests/test_contract.py` checks each rule against an in-memory Hub
(`tests/fake_bucket.py`) that follows the Hub's own listing rules. A rule marked **gap** is one
the code breaks; its test is a strict xfail until the fix lands. No gap is open.

## 1. Addressing

- A bucket reference is `ns/bucket[/prefix]`, `hf://buckets/ns/bucket[/prefix]` or a Hub web
  URL (`https://huggingface.co/buckets/ns/bucket/tree/prefix`). Anything else raises
  `ValueError`, including an `hf://` URI of a model or dataset.
- A prefix is a string prefix of the key, as on the Hub. A trailing `/` limits it to one
  directory (`a/b/` lists `a/b/x`, not `a/bc/y`); without the slash, siblings match too.
  The same rule holds for `prefix=` and for a prefix inside the bucket reference.
- `prefix=` joins below a prefix inside the reference, as directories: `ns/b/out` with
  `prefix="part-"` lists `out/part-*`. `prefix=""` or `None` adds nothing.
- Globs (`include`, `exclude`) match the whole key from the bucket root, with fsspec's rules,
  the ones `HfFileSystem.glob` uses: `*` stays inside one path segment, `**` as a whole segment
  crosses `/`, `?` is one character and `[...]` a character class. `**` inside a name
  (`**.jp2`) raises `ValueError`; write `**/*.jp2`. (`hf buckets sync --include` uses
  `fnmatch`, where `*` also crosses `/`.)
- Listings return full keys from the bucket root, sorted, files only. `start_after` skips
  keys up to and including it. `limit` keeps the first `limit` keys in the Hub's listing
  order, which is lexical today but not documented (issue #9).
- `put_files(pairs, "ns/bucket/prefix")` writes each key under that prefix, joined as a
  directory.

## 2. Temporary files

`batched_files` deletes a batch's files when the loop moves past it. It also deletes them when
the loop body raises, when the caller stops early, and for batches downloaded ahead
(`prefetch`). `tests/test_batched_files.py` covers each `prefetch` level.

At most `prefetch + 1` batches are on disk at once: the one you are processing and `prefetch`
downloading ahead. `prefetch` is also the number of download threads (at least one). Downloads
always run in a thread, so Ctrl-C waits for the download in flight, then deletes its files.
A second Ctrl-C during that wait stops the clean-up, and the batches still downloading may
stay on disk.

"Stops early" means the generator is closed. A `for` loop over `batched_files(...)` closes it
when the loop ends. A caller that keeps the generator in a variable must call `.close()`
(or use `contextlib.closing`); until then the current and lookahead batches stay on disk.

## 3. Resume (`Bag`)

- `to_bucket(out, done=".md")` names the output that marks an input done: an input `k` is done
  when `k + ".md"` exists under `out` (`pages/p1.jp2` → `pages/p1.jp2.md`). The lookup is exact,
  so no other output can mark `k` done (`pages/p10.md` never marks `pages/p1`). `compute()`
  processes only inputs that are not done; `compute(resume=False)` processes all of them.
  With nothing to do, `compute()` returns without running `setup()`.
- `compute(resume=True)` without `done=` raises `ValueError` before anything is listed.
- An input whose batch wrote no `k + done` output is logged as a warning, once per `compute()`
  or `take()`. Resume cannot mark it done, so it runs again on every re-run.
- `take(n)` runs the plan on the first `n` inputs and writes nothing. `setup()` runs at most
  once per `compute()` or `take()`.

## 4. Failures (`Bag.compute`)

| What fails | What happens |
|---|---|
| The map function raises | `compute()` raises that error. Earlier batches' outputs stay. |
| A download fails | `compute()` raises that error. Earlier batches' outputs stay. |
| An upload fails | `compute()` raises that error. Outputs that landed before it count as done.* |

In every row the temporary files are deleted, and a re-run processes exactly the inputs whose
`done` output is missing. Hypothesis checks this over random mixes of failures (`tests/test_fault_fuzz.py`).

\* The server applies the files of an upload that succeed and reports the rest.
huggingface_hub 2.x raises `BucketBatchError` on that report. 1.x checked only the HTTP status,
so `compute()` went on, and the missing outputs were redone on the next run.

A file that is listed but gone by download time: with string keys
(`batched_files(keys=[...])`) the hub skips it with a warning, and bucketbag leaves it out of
the batch. `Bag` passes `BucketFile` objects, which the hub downloads by content hash without
an existence check. What happens then is not verified live; the hub source suggests the download
raises (`EntryNotFoundError`) at least when the gone file comes first in the batch.

## 5. Arguments

- `token=` reaches every hub call.
- `batch_size` caps a batch's file count and `max_bytes` its total size; a single file larger
  than `max_bytes` forms its own batch. `max_bytes` with string keys raises `ValueError`:
  string keys carry no size.
- Settings that cannot work raise before anything is listed. `n` (`batch_size` in `Bag`) and
  `prefetch` must be whole numbers; `max_bytes` may be any finite number, such as `4e9`. A bool,
  nan or inf raises `TypeError`. At least one of `n` and `max_bytes` must be set, each at least
  1, and `prefetch` must be 0 or more (`ValueError`). `Bag.map_batches` checks its bounds when
  the plan is built.
- The map function is called as `fn(items)`, or `fn(items, ctx)` when `setup=` is given. Its
  signature is not inspected.
- `completed_keys` skips a shard it cannot read, with a warning. A readable shard without
  `column` raises `ValueError`, so a wrong `column=` cannot make resume rerun everything.

## 6. Run report (`Bag`)

- `compute()` returns `BagStats` with the time spent waiting for the next batch (`wait_s`: its
  download, plus deleting the last batch's files), in the map function (`map_s`) and writing
  outputs (`upload_s`). The three exclude `setup()` and
  listing; `seconds` includes them. `bound_by` names the largest of the three and `hint` the
  setting to change. When no batch ran, `bound_by` is `None` and `hint` is `""`.
  `print(stats)` shows the split and the verdict.
- `compute()` logs its settings when it starts and its `BagStats` when it ends. `compute()` and
  `take()` log a progress line after a batch, at most every 30 seconds: inputs done out of the
  total, the rate, the time left and the split. A long batch delays its line. All go to the `bucketbag` logger at INFO, and print to stderr as
  `[bucketbag] …` lines until the caller configures logging; then the caller's handlers get
  them.
- `batched_files` warns before the first download when the largest `prefetch + 1` batches
  would not fit in the scratch dir's free space. String keys carry no size, so they never warn.

## 7. Process-wide xet settings

- Importing bucketbag sets `HF_XET_HIGH_PERFORMANCE=1` and
  `HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=32`, each only when the caller has not set it.
  `BUCKETBAG_NO_XET_TUNE=1` sets neither. Both apply to every Hugging Face download in the
  process.
- `boost(file_concurrency=k)` replaces bucketbag's import-time values, never values the caller
  exported.
- `batched_files` logs both values, and who set each, at INFO once per process, before its first
  download. When xet had already started before bucketbag was imported (the process downloaded
  from the Hub first), xet may keep its own values; the line then says so, at WARNING.

`tests/test_xet_env.py` checks each rule.
