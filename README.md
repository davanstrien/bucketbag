# bucketbag

**Experimental.** A small, [toolz](https://toolz.readthedocs.io/)-style library for batch-processing
**large collections of files in [Hugging Face buckets](https://huggingface.co/docs/hub/en/storage-buckets)** — an experiment in what bucket-scale batch work
wants as an interface:

- **Bounded scratch** — download, process, delete, repeat. Peak footprint ≈ the download budget (default 2 GB, or three batches when batches are larger; at most half the free space), never the corpus.
- **Resume** — re-runs skip work already done (long [HF Jobs](https://huggingface.co/docs/hub/en/jobs) can stop and restart).
- **Zero cleanup code** — temp files removed as the loop advances, even on exceptions.
- **Throughput defaults** — prefetch overlaps download with compute; importing bucketbag raises xet's concurrent-file limit to 32 (see [Process-wide settings](#process-wide-settings)).

Mounting a bucket in a Job (`hf jobs run -v hf://buckets/...`) is the simplest way to use one; bucketbag explores the
**explicit API path** for batch workloads where footprint, resume, and concurrency should be under
your control. Not faster than the raw API (same transport), and not a framework.

## Install

```toml
# /// script
# dependencies = ["bucketbag==0.4.0"]
# ///
```

Or `pip install bucketbag`. Pin a version: bucketbag is experimental and 0.x releases break.
[CHANGELOG.md](https://github.com/davanstrien/bucketbag/blob/main/CHANGELOG.md) says how to move from 0.3.

## The read loop

```python
from bucketbag import batched_files

for batch in batched_files("ns/my-bucket", include="**/*.jp2", max_bytes=4 * 2**30):
    for it in batch:                 # LoadedItem, already on local disk
        work(it.path)                # or it.bytes / it.image / it.text() / it.json()
    # ↑ this batch's files are deleted as the loop advances
```

Default scratch dir is `/dev/shm` (**RAM tmpfs, not disk**). Downloads run ahead of your loop
within a byte budget: 2 GB by default, or three batches when batches are larger than about
670 MB, and at most half the scratch dir's free space. On a small machine pass a hard budget,
`prefetch_bytes=256e6` (or `0` for one download call at a time), or `dir=` on real disk. One call
always downloads ahead, so a batch larger than a hard budget means about two batches on disk.
One rule: don't keep a `LoadedItem`/`.path` past its batch.

## Writing + resume

```python
from bucketbag import iter_keys, batched_files, put_files

done = {k.removesuffix(".md") for k in iter_keys(OUT, include="**/*.md")}
keys = [k for k in iter_keys(SRC, include="**/*.jp2", objects=True) if k.path not in done]
for batch in batched_files(SRC, keys=keys, max_bytes=4 * 2**30):
    put_files([(it.key + ".md", ocr(it)) for it in batch], OUT)   # one API call per batch
```

`iter_keys(..., objects=True)` yields `BucketFile` objects (with `.size`) instead of key
strings — pass those to `batched_files(keys=…, max_bytes=…)` so the byte bound can be honored.
Bare string keys carry no size, so **`max_bytes` + string keys raises `ValueError`** (fail fast
rather than silently run unbounded against the default RAM-tmpfs scratch — an OOMKill, not a
disk-full). Let `batched_files` do the listing (omit `keys=`) and it keeps the sizes for you.

Parquet outputs are your tool's job (polars / pyarrow straight to the bucket);
`completed_keys(OUT)` reads a done-set back from a `__source_key` column (pyarrow in *your* deps).

## `Bag` — the loop as a lazy plan (experimental)

```python
from bucketbag import Bag

bag = (Bag.from_bucket(SRC, include="**/*.jp2")
         .map_batches(ocr_batch, batch_size=32, setup=load_model)   # fn(items[, ctx]) -> [(key, content)]
         .to_bucket(OUT, done=".md"))                                  # input k is done once k + ".md" exists
bag.take(3)        # dev peek: outputs only, nothing written
stats = bag.compute()  # list -> skip done -> download -> fn -> put_files; kill + re-run safe
print(stats)  # for example:
# 476 processed (+24 skipped as done) in 15 batches -> 952 outputs, 3712.4s
# time: download wait 1%, map 98%, upload 1%; bound by map. The map function is the limit: ...
```

`stats.bound_by` (`"download"`, `"map"` or `"upload"`) and `stats.hint` tell a person or an
agent what to tune next. `compute()` also logs its settings, a progress line after batches
(at most every 30 s) and the same report at the end. They print to stderr as `[bucketbag] …`
lines, so they show up in `hf jobs logs`, until your script configures logging; then your
handlers get them.

Resume: `done=".md"` names the output that marks an input done, and the lookup is exact
(`pages/p10.md` never marks `pages/p1` done). `compute()` runs in-process today; fanning the same
plan across HF Jobs (one shard per job) is the next layer.

To follow a bucket that another job is still filling, re-run the plan. Each pass lists both
buckets again and skips what is done; `setup()` runs only when a pass has work:

```python
import time

while producer_running():
    bag.compute()
    time.sleep(60)
bag.compute()  # once more: files may have landed during the last sleep
```

[CONTRACT.md](https://github.com/davanstrien/bucketbag/blob/main/CONTRACT.md) states what bucketbag promises; the tests check each rule.

## API

| | |
| --- | --- |
| `batched_files(bucket, *, keys, include, exclude, n=20, max_bytes, dir, prefetch_bytes=None, ...)` | download batches → yield `list[LoadedItem]` → auto-delete |
| `iter_keys(bucket, *, prefix, include, exclude, start_after, limit, token)` | list + glob-filter + sort keys (no download) |
| `put_files(pairs, out_bucket)` / `put_bytes` / `put_text` | write raw objects; `put_files` batches many per call |
| `completed_keys(out_bucket, *, prefix, column="__source_key")` | done-set from parquet outputs |
| `boost(*, file_concurrency=32)` | change xet's concurrent-file limit (bucketbag sets 32 at import) |
| `LoadedItem` | `.key` `.path` + lazy `.bytes` `.image` `.text()` `.json()` |
| `Bag.from_bucket(...).map_batches(fn).to_bucket(out, done=".md")` + `.take(n)` / `.compute()` | the loop as a lazy plan |

`bucket` = `"ns/bucket"`, `"ns/bucket/prefix"`, `"hf://buckets/ns/bucket/prefix"`, or a Hub web URL.
`prefix=` joins below a prefix inside `bucket`, as a directory.
Globs follow fsspec's rules (as in `HfFileSystem.glob`): `*` within a path segment, `**` across `/`.
`prefix` is a string prefix and the trailing slash matters: `prefix="a/b/"` lists the
directory `a/b` only; `prefix="a/b"` also matches siblings like `a/bc/y`. The prefix is sent
verbatim to the Hub. Bucketbag also checks each returned key against it as a cheap safeguard
against over-matching listings, including pagination regressions.

## Performance

HF Jobs `cpu-upgrade`, 2026-11-05: throughput as a fraction of **one** `HfApi.download_bucket_files`
call for all of a run's files (an upper bound for bounded scratch; it holds every file on disk).
Median of paired rounds, each version in its own process, default settings (`n=20, prefetch=2`;
`n=2` for 45 MB, `n=4` for 0.5 GB):

| files | 0.4.0 | + xet limit 32 | + grouped calls |
| --- | --- | --- | --- |
| 19 KB jpg | 0.26 | 0.36 | **0.90** |
| 0.8 MB jp2 | 0.30 | 0.52 | **0.90** |
| 2.5 MB jpg | 0.33 | 0.76 | **0.96** |
| 45 MB flac | 0.53 | 0.79 | **0.80** |
| 0.5 GB parquet (CDN warm) | 1.20 | 1.27 | 1.21 |

Two changes do it. xet allows 8 concurrent file downloads per process by default, shared by every
call; bucketbag sets 32 at import. And every hub call costs about 0.4 s before data moves (a
metadata request and a new xet download group), so small batches now share a call of about
64 MB. Grouped calls alone gain nothing while xet's limit is 8. The first batch of tiny files
waits for its whole call (about 1.5 s for 19 KB files, against 0.5 s before).

This version bounds the downloads ahead by bytes, not by a count of batches, with up to 8
download threads. Against grouped calls with `prefetch=2`, in the same rounds:

| files | `prefetch=2` | byte budget (default) |
| --- | --- | --- |
| 19 KB jpg | 0.95 | **1.00** |
| 0.8 MB jp2 | 0.94 | 0.95 |
| 2.5 MB jpg | 0.86 | **1.04** |
| 45 MB flac | 0.70 | **0.80** |
| 220 MB parquet | 1.00 | 1.07 |
| 0.5 GB parquet, 4 per batch (two runs) | 1.24, 0.89 | 1.08, 0.91 |

A flat 2 GB budget gave 0.79 and 0.96 on the 0.5 GB rows: it kept one 2.2 GB batch ahead, where
0.4.0 kept two. The default therefore also holds three of the largest call. With batches that
large the first batch arrives later (about 8 s against 5 s), since three calls share the
bandwidth; pass `prefetch=2` if that matters. A hard budget of `prefetch_bytes=256e6` gave
0.78–1.01 on the same sizes. Raise the budget when `Bag`'s run report says downloads are the
limit.

### Process-wide settings

Importing bucketbag sets two xet environment variables, unless you have set them yourself:
`HF_XET_HIGH_PERFORMANCE=1` and `HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=32`. They apply to
**every Hugging Face download in the process**, not only bucketbag's: a model download in the same
script also runs up to 32 files at once. xet still caps its download buffer memory (half of usable
memory in high-performance mode, cgroup-aware). The first `batched_files` call logs the values and
who set them:

```text
[bucketbag] xet settings for this process: HF_XET_HIGH_PERFORMANCE=1 (set by bucketbag), HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS=32 (set by bucketbag). ...
```

xet reads them once, when its first download starts, so **import bucketbag before the first Hugging
Face download** in the process; otherwise the log line warns that xet may keep its own values. To
change them, export your own values before importing bucketbag, call `boost(file_concurrency=k)`
before the first download, or set `BUCKETBAG_NO_XET_TUNE=1` to keep xet's defaults.

### Decode, not transport

If a pipeline is slow, time the download and the decode separately before tuning transport
(`examples/decode_probe.py` does this). The transport is usually not the gate: on ~0.7 MB page
images, `batched_files` moves 40 files in ~1 s, and decoding those 40 takes ~70 s. Three things
to check in the decode step:

- **Threads may not help.** Some Pillow decoders hold the GIL (JPEG 2000 does): 8 threads gave the
  same speed as 1. A process pool did scale.
- **Decode smaller.** Formats with resolution levels (JPEG 2000: `im.reduce`; JPEG: `im.draft()`)
  can decode at 1/4 or 1/8 the pixels for a fraction of the time — usually all a classifier needs.
- **Check the reduced size loads.** Pillow's JPEG 2000 `reduce>=2` fails with "broken data stream"
  when its size rounding disagrees with OpenJPEG's; `safe_reduce()` in the example picks the
  largest level that loads.

## Not in scope

DAGs, shuffles, custom formats, return values to a driver. Stages couple through bucket keys; a
global reduce is a one-off polars job. Jobs fan-out is planned as a thin layer on `Bag`.
