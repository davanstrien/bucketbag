# bucketbag

A minimal, [toolz](https://toolz.readthedocs.io/)-style helper for batched, bounded-memory access to
**Hugging Face bucket** files. Not a framework — just a few composable functions you can drop into any
script (including [`uv`](https://docs.astral.sh/uv/) PEP 723 scripts).

`huggingface_hub` already has the raw pieces (`list_bucket_tree`, `download_bucket_files`,
`batch_bucket_files`). What you end up rewriting in every script is the *batch-download → process →
delete → repeat* loop, without blowing up disk or leaking temp files. `bucketbag` is that loop.

## Install

```toml
# in a uv script's PEP 723 header
# /// script
# dependencies = ["bucketbag @ git+https://github.com/davanstrien/bucketbag"]
# ///
```

or `uv add "bucketbag @ git+https://github.com/davanstrien/bucketbag"` / `pip install -e .` for dev.

## The headline verb: `batched_files`

Think `toolz.partition_all` for **bucket files** — except each batch is downloaded to a temp dir
(`/dev/shm` by default) and **deleted before the next**, so disk stays bounded. It is **file-type
agnostic**.

```python
from bucketbag import batched_files

for batch in batched_files("davanstrien/my-bucket", include="images/**/*.jpg", n=20):
    for it in batch:                 # it: LoadedItem, already on local disk
        do_something(it.path)        # or it.bytes / it.image / it.text() / it.json()
    # ↑ this batch's files are auto-deleted as the loop advances (bounded disk)
```

`prefetch` overlaps the next download with your current processing (raising disk high-water to
≈`(prefetch+1)*n` files):

```python
for batch in batched_files("davanstrien/my-bucket", include="**/*.jp2", n=32, prefetch=2):
    ...
```

## Resume (your loop, your rules)

```python
from bucketbag import iter_keys, batched_files, completed_keys, write_parquet

SRC, OUT = "davanstrien/my-bucket", "davanstrien/my-out"
done = completed_keys(OUT)                                  # set of __source_key already written
keys = [k for k in iter_keys(SRC, include="images/**/*.jp2") if k not in done]

for batch in batched_files(SRC, keys=keys, n=32, prefetch=2):
    rows = [{"__source_key": it.key, "n_bytes": len(it.bytes)} for it in batch]
    write_parquet(rows, OUT, f"part-{batch[0].key.replace('/', '_')}.parquet")
```

## Performance

Measured with `examples/bench.py` on real BHL page images (`.jp2`, ~0.8 MB), **cold + disjoint**
(every config reads its own fresh slice — matches the real "each file read once" workload and avoids
cache inflation), replicated, median of N. Numbers below are an l4x1 Job (n=300 unless noted); they're
moderately noisy (±10–20%), so read them as ranges, not decimals.

**Default transport (`HF_XET_HIGH_PERFORMANCE=1`, on by default):**

| method | img/s | notes |
| --- | --- | --- |
| FUSE mount (32 threads) | ~22 | ~4–5× slower — avoid |
| `bucketbag` (prefetch=2–4) | ~85–110 | ties the alternatives; bounded disk + cleanup + resume |
| raw `download_bucket_files` | ~85–105 | one big call; no bounded disk/cleanup |
| `HfFileSystem` (32 threads) | ~90–110 | signed-URL range reads (signed-URL expiry caveat) |

So on the **default** transport `bucketbag` is **competitive, not faster** — its value is bounded disk,
auto-cleanup, resume, and overlap-with-compute, at ~no throughput cost.

**With `bucketbag.boost()` (raises xet's per-process file-download cap; best for many small files):**

| method | img/s |
| --- | --- |
| `bucketbag` (prefetch=2–4) + `boost()` | **~200–270** |
| raw `download_bucket_files` + same env | ~210–260 |
| `HfFileSystem` (unaffected — bypasses xet) | ~110 |

Raising xet's concurrent-file cap (default 8) **~2.5×'d** throughput on small files and made the HfApi
path clearly beat `HfFileSystem`. It's a single env var — **no Rust / native code needed** (a Rust
`xet-core` wrapper would reimplement what hf-xet already does, for the same result). Note: best for
*small* files; for *large* files leave it default (each file already fans out into many range GETs, so a
high file-cap over-subscribes). `HF_XET_NUM_CONCURRENT_RANGE_GETS` is the knob for *large* files.

Cross-stage tip: the xet chunk cache is **off by default** (correct for read-once workloads). If a later
stage re-reads files an earlier stage fetched, set `HF_XET_CHUNK_CACHE_SIZE_BYTES` (10 GB+) to skip the
re-fetch.

## API

| function | what it does |
| --- | --- |
| `batched_files(bucket, *, keys, include, exclude, n, max_bytes, dir, prefetch, max_workers, ...)` | download files in batches to a temp dir, yield `list[LoadedItem]`, auto-delete each batch |
| `iter_keys(bucket, *, include, exclude, start_after, limit, ...)` | list + glob-filter + sort keys (cheap, no download) |
| `completed_keys(out_bucket, *, prefix, column)` | set of already-done keys from parquet outputs (resume) |
| `write_parquet(rows, out_bucket, key)` | one-shot: write a list of dicts as a parquet object |
| `boost(*, file_concurrency=32, high_performance=True)` | raise xet download concurrency for many-small-files (~2.5×); call before first download |
| `LoadedItem` | `key` + local `path` + lazy `.bytes` / `.image` / `.text()` / `.json()` |
| `partition_all` | re-exported from `toolz` (pure index chunking, no I/O) |

Globs use proper path semantics: `*` matches within a segment, `**` crosses `/`, `**/` matches zero or
more directories. A `bucket` may be `"ns/bucket"`, `"ns/bucket/prefix"`, or `"hf://buckets/ns/bucket/prefix"`.

**Bounding the footprint:** pass `max_bytes` (e.g. `max_bytes=4 * 2**30`, optionally with `n=None`) to cap
each batch by *total bytes* instead of file count — disk high-water is then ≈ `(prefetch + 1) * max_bytes`
regardless of file size. **Throughput:** `HF_XET_HIGH_PERFORMANCE=1` is set by default (opt out with
`BUCKETBAG_NO_XET_TUNE=1`); for many small files call `bucketbag.boost()` for the ~2.5× (see Performance).

## Scope

`bucketbag` is intentionally small and **pure Python** — no native/Rust code (the one lever that matters,
xet's file-download concurrency, is a single env var; see Performance). Jobs fan-out (re-invoke-self
across many HF Jobs with disjoint shard keys, `_SUCCESS` markers, poll/collect) is **deliberately out of
scope for now** — if it lands later it will be a separate thin `bucketbag.jobs` module built on these
helpers, not a framework.

Possible future: adaptive batch sizing (pick `n`/`max_bytes` from observed file sizes to hit a target
footprint) — the natural "auto batch size" successor to today's explicit `max_bytes`.
