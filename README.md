# bucketbag

A minimal, [toolz](https://toolz.readthedocs.io/)-style helper for **batched, bounded-memory** reads of
**Hugging Face bucket** files. Not a framework — a few composable functions for the
*download-a-batch → process → delete → repeat* loop you'd otherwise rewrite in every script.

## Install

```toml
# uv script PEP 723 header
# /// script
# dependencies = ["bucketbag @ git+https://github.com/davanstrien/bucketbag"]
# ///
```
(or `uv add "bucketbag @ git+https://github.com/davanstrien/bucketbag"` / `pip install -e .`)

## `batched_files` — the one verb

`partition_all` for bucket files: it lists/downloads a batch to a temp dir (`/dev/shm`), yields it, and
**deletes it before the next** — so disk stays bounded. **File-type agnostic.** Cleanup is automatic
(even on exception) — you never touch temp files.

```python
from bucketbag import batched_files

# Bound each batch by SIZE (recommended) — predictable footprint whatever the file sizes:
for batch in batched_files("davanstrien/my-bucket", include="**/*.jp2", max_bytes=4 * 2**30):
    for it in batch:                 # LoadedItem, already on local disk
        work(it.path)                # or it.bytes / it.image / it.text() / it.json()
    # ↑ this batch's files are deleted as the loop advances — nothing to clean up

# …or bound by file count instead:
for batch in batched_files("davanstrien/my-bucket", include="**/*.jp2", n=32):
    ...
```

Disk high-water ≈ `(prefetch + 1) × max_bytes`. `prefetch=2` (default) overlaps downloads with your work.
The only rule: **don't keep a `LoadedItem`/`.path` past its batch** — the file is already gone.

## Resume (your loop, your rules)

```python
from bucketbag import iter_keys, batched_files, completed_keys, write_parquet

done = completed_keys(OUT)                                    # __source_key values already written
keys = [k for k in iter_keys(SRC, include="**/*.jp2") if k not in done]
for batch in batched_files(SRC, keys=keys, max_bytes=4 * 2**30):
    rows = [{"__source_key": it.key, **work(it)} for it in batch]
    write_parquet(rows, OUT, f"part-{batch[0].key.replace('/', '_')}.parquet")
```

## API

| | |
| --- | --- |
| `batched_files(bucket, *, keys, include, exclude, n=20, max_bytes, dir, prefetch=2, max_workers, start_after, limit, token)` | download batches → yield `list[LoadedItem]` → auto-delete |
| `iter_keys(bucket, *, prefix, include, exclude, start_after, limit, token)` | list + glob-filter + sort keys (no download) |
| `completed_keys(out_bucket, *, prefix, column="__source_key", token)` | set of done keys, for resume |
| `write_parquet(rows, out_bucket, key, *, token)` | write a list of dicts as one parquet object |
| `boost(*, file_concurrency=32, high_performance=True)` | raise xet download concurrency (~2.5× on small files) |
| `LoadedItem` | `.key` `.path` + lazy `.bytes` `.image` `.text()` `.json()` |
| `partition_all` | re-exported from `toolz` |

`bucket` = `"ns/bucket"`, `"ns/bucket/prefix"`, or `"hf://buckets/ns/bucket/prefix"`. Globs: `*` within a
path segment, `**` across `/`. `n=None` + `max_bytes` gives pure size-based batches.

## Performance

Cold + disjoint, replicated (`examples/bench.py`, l4x1, ~0.8 MB jp2; ranges, ±10–20%):

| | default xet | + `boost()` |
| --- | --- | --- |
| `bucketbag` (prefetch 2–4) | ~85–110 img/s | **~200–270 img/s** |
| raw `download_bucket_files` | ~85–105 | ~210–260 |
| `HfFileSystem` (32 threads) | ~90–110 | ~110 (bypasses xet) |
| FUSE mount | ~22 (avoid) | — |

On the default transport bucketbag is **competitive, not faster** — the win is bounded disk + cleanup +
resume + overlap-with-compute at ~no cost. The one real throughput lever is xet's concurrent-file cap
(default 8): `boost()` raises it for **~2.5× on small files** — a single env var, **no Rust**. Skip
`boost()` for *large* files (it would over-subscribe). `HF_XET_HIGH_PERFORMANCE=1` is on by default (opt
out `BUCKETBAG_NO_XET_TUNE=1`); for cross-stage re-reads, enable `HF_XET_CHUNK_CACHE_SIZE_BYTES`.

## Scope

Intentionally small and **pure Python**. Out of scope (for now): Jobs fan-out — would be a separate thin
`bucketbag.jobs` module on top of these helpers. Possible future: adaptive "auto batch size".
