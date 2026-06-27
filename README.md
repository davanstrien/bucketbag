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

## API

| function | what it does |
| --- | --- |
| `batched_files(bucket, *, keys, include, exclude, n, dir, prefetch, max_workers, ...)` | download files in batches to a temp dir, yield `list[LoadedItem]`, auto-delete each batch |
| `iter_keys(bucket, *, include, exclude, start_after, limit, ...)` | list + glob-filter + sort keys (cheap, no download) |
| `completed_keys(out_bucket, *, prefix, column)` | set of already-done keys from parquet outputs (resume) |
| `write_parquet(rows, out_bucket, key)` | one-shot: write a list of dicts as a parquet object |
| `LoadedItem` | `key` + local `path` + lazy `.bytes` / `.image` / `.text()` / `.json()` |
| `partition_all` | re-exported from `toolz` (pure index chunking, no I/O) |

Globs use proper path semantics: `*` matches within a segment, `**` crosses `/`, `**/` matches zero or
more directories. A `bucket` may be `"ns/bucket"`, `"ns/bucket/prefix"`, or `"hf://buckets/ns/bucket/prefix"`.

For best throughput set `HF_XET_HIGH_PERFORMANCE=1` in your environment (xet transport).

## Scope

`bucketbag` is intentionally small. Jobs fan-out (re-invoke-self across many HF Jobs with disjoint shard
keys, `_SUCCESS` markers, poll/collect) is **deliberately out of scope for now** — if it lands later it
will be a separate thin `bucketbag.jobs` module built on these helpers, not a framework.
