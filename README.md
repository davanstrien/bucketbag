# bucketbag

**Experimental.** A small, [toolz](https://toolz.readthedocs.io/)-style library for batch-processing
**large collections of Hugging Face bucket files** — an experiment in what bucket-scale batch work
wants as an interface:

- **Bounded scratch** — download a batch, process, delete, repeat. Peak footprint ≈ `(prefetch + 1) × max_bytes`, never the corpus.
- **Resume** — re-runs skip work already done (long [HF Jobs](https://huggingface.co/docs/hub/en/jobs) can stop and restart).
- **Zero cleanup code** — temp files removed as the loop advances, even on exceptions.
- **Throughput knobs** — prefetch overlaps download with compute; `boost()` raises xet's small-file concurrency.

Mounting a bucket (`-v hf://buckets/...`) is the simplest way to use one; bucketbag explores the
**explicit API path** for batch workloads where footprint, resume, and concurrency should be under
your control. Not faster than the raw API (same transport), and not a framework.

## Install

```toml
# /// script
# dependencies = ["bucketbag @ git+https://github.com/davanstrien/bucketbag@v0.3.0"]
# ///
```

Pin a tag — HEAD moves.

## The read loop

```python
from bucketbag import batched_files

for batch in batched_files("ns/my-bucket", include="**/*.jp2", max_bytes=4 * 2**30):
    for it in batch:                 # LoadedItem, already on local disk
        work(it.path)                # or it.bytes / it.image / it.text() / it.json()
    # ↑ this batch's files are deleted as the loop advances
```

Default scratch dir is `/dev/shm` (**RAM tmpfs, not disk**) — size `(prefetch + 1) × max_bytes`
against memory, or pass `dir=`. One rule: don't keep a `LoadedItem`/`.path` past its batch.

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
         .to_bucket(OUT))
bag.take(3)        # dev peek: outputs only, nothing written
bag.compute()      # list -> skip done -> download -> fn -> put_files; kill + re-run safe
```

Resume contract: an input is done iff an output key **starts with** its source key. `compute()`
runs in-process today; fanning the same plan across HF Jobs (one shard per job) is the next layer.

## API

| | |
| --- | --- |
| `batched_files(bucket, *, keys, include, exclude, n=20, max_bytes, dir, prefetch=2, ...)` | download batches → yield `list[LoadedItem]` → auto-delete |
| `iter_keys(bucket, *, prefix, include, exclude, start_after, limit, token)` | list + glob-filter + sort keys (no download) |
| `put_files(pairs, out_bucket)` / `put_bytes` / `put_text` | write raw objects; `put_files` batches many per call |
| `completed_keys(out_bucket, *, prefix, column="__source_key")` | done-set from parquet outputs |
| `boost(*, file_concurrency=32)` | raise xet download concurrency (~2.5× on small files) |
| `LoadedItem` | `.key` `.path` + lazy `.bytes` `.image` `.text()` `.json()` |
| `Bag.from_bucket(...).map_batches(fn).to_bucket(out)` + `.take(n)` / `.compute()` | the loop as a lazy plan |

`bucket` = `"ns/bucket"`, `"ns/bucket/prefix"`, or `"hf://buckets/ns/bucket/prefix"`.
Globs: `*` within a path segment, `**` across `/`.
`prefix` is a string prefix and the trailing slash matters: `prefix="a/b/"` lists the
directory `a/b` only; `prefix="a/b"` also matches siblings like `a/bc/y`. (The Hub honors the
slash on the first page of a listing but its pagination links drop it, so bucketbag re-applies
the prefix client-side — since 0.3.1.)

## Performance

Cold + disjoint, replicated (`examples/bench.py`, l4x1, ~0.8 MB jp2; ±10–20%):

| | default xet | + `boost()` |
| --- | --- | --- |
| `bucketbag` (prefetch 2–4) | ~85–110 img/s | **~200–270 img/s** |
| raw `download_bucket_files` | ~85–105 | ~210–260 |
| `HfFileSystem` (32 threads) | ~90–110 | ~110 (bypasses xet) |
| FUSE mount (same workload) | ~22 | — |

Competitive with the raw API by design; the win is the interface. The one real throughput lever is
xet's concurrent-file cap (default 8) — `boost()` raises it, ~2.5× on small files. Skip it for
large files (over-subscription).

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
