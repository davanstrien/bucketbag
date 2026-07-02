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

Scratch high-water ≈ `(prefetch + 1) × max_bytes` — and the default scratch dir is `/dev/shm`
(**RAM tmpfs, not disk**), so size that against available memory or pass `dir=` for real disk.
`prefetch=2` (default) overlaps downloads with your work.
The only rule: **don't keep a `LoadedItem`/`.path` past its batch** — the file is already gone.

## Resume (your loop, your rules)

Resume is just: list what's done, skip it, pass the rest as `keys=`. Two common shapes:

```python
from bucketbag import iter_keys, batched_files, put_files

# Per-file outputs (one .md per page): what's done = which outputs exist.
done = {k.removesuffix(".md") for k in iter_keys(OUT, include="**/*.md")}
keys = [k for k in iter_keys(SRC, include="**/*.jp2") if k not in done]
for batch in batched_files(SRC, keys=keys, max_bytes=4 * 2**30):
    put_files([(it.key + ".md", ocr(it)) for it in batch], OUT)   # one API call per batch
```

```python
from bucketbag import completed_keys   # parquet outputs: needs pyarrow in YOUR deps

done = completed_keys(OUT)             # __source_key values already written to *.parquet
```

Writing parquet itself is your tool's job (polars / pyarrow straight to the bucket) — bucketbag
only writes raw objects (`put_files` / `put_bytes` / `put_text`).

## `Bag` — the same loop as a plan (experimental)

A lazy, dask.bag-style chain over the helpers above — listing, resume, bounded download,
batched writes in one shape:

```python
from bucketbag import Bag

bag = (Bag.from_bucket(SRC, include="**/*.jp2")
         .map_batches(ocr_batch, batch_size=32, setup=load_model)   # fn(items[, ctx]) -> [(key, content)]
         .to_bucket(OUT))
bag.take(3)        # dev peek: outputs for the first 3 items, nothing written
bag.compute()      # list -> skip done -> download batches -> fn -> put_files; kill + re-run safe
```

Resume contract: an input is done iff an output key **starts with** its source key (so emit
`it.key + ".md"`-style keys). `compute()` runs in-process for now — Jobs fan-out
(`shards -> one Job each`) is the planned next layer on this same plan object.

## API

| | |
| --- | --- |
| `batched_files(bucket, *, keys, include, exclude, n=20, max_bytes, dir, prefetch=2, max_workers, start_after, limit, token)` | download batches → yield `list[LoadedItem]` → auto-delete |
| `iter_keys(bucket, *, prefix, include, exclude, start_after, limit, token)` | list + glob-filter + sort keys (no download) |
| `completed_keys(out_bucket, *, prefix, column="__source_key", token)` | set of done keys from parquet outputs (needs pyarrow in your deps) |
| `put_files(pairs, out_bucket, *, encoding, token)` / `put_bytes` / `put_text` | write raw objects; `put_files` batches many in one call |
| `boost(*, file_concurrency=32, high_performance=True)` | raise xet download concurrency (~2.5× on small files) |
| `LoadedItem` | `.key` `.path` + lazy `.bytes` `.image` `.text()` `.json()` |
| `Bag.from_bucket(...).map_batches(fn, ...).to_bucket(out)` + `.take(n)` / `.compute()` | the loop as a lazy plan (see above) |
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
