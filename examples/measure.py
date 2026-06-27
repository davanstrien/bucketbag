# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bucketbag @ git+https://github.com/davanstrien/bucketbag",
# ]
# ///
"""Minimal bucketbag example: measure pages in a bucket, bounded + resumable.

An I/O-only pass over Biodiversity Heritage Library page images: download in batches to a temp
dir (auto-cleaned), record one row per page, write parquet back to a bucket, and resume by
skipping pages already done. No image decoding — this is about the read/batch/cleanup loop.

    uv run examples/measure.py                 # ~300 pages, prefetch=2
    uv run examples/measure.py --limit 600 --n 64 --prefetch 3
    uv run examples/measure.py                 # run again -> skips everything (resume)

Set HF_XET_HIGH_PERFORMANCE=1 for best throughput (done below).
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import create_bucket  # noqa: E402

from bucketbag import batched_files, boost, completed_keys, iter_keys, write_parquet  # noqa: E402

SRC = "davanstrien/finebooks-bhl-pilot"
OUT = "davanstrien/bucketbag-mvp-out"


def peak_rss_mb() -> float:
    """Peak resident set size in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1024


def dir_size_mb(path: Path) -> float:
    """Total size of files under `path` in MB (the live download footprint)."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=300, help="number of pages to process")
    ap.add_argument("--n", type=int, default=32, help="files per batch")
    ap.add_argument("--prefetch", type=int, default=2, help="batches to download ahead")
    ap.add_argument("--include", default="images/**/*.jp2", help="glob to select pages")
    args = ap.parse_args()

    scratch = Path(os.environ.get("BUCKETBAG_DIR", "/dev/shm"))
    if not (scratch.is_dir() and os.access(scratch, os.W_OK)):
        scratch = Path("/tmp")
    scratch = scratch / "bucketbag-measure"
    scratch.mkdir(parents=True, exist_ok=True)

    # BHL pages are small (~1 MB), so raise xet's file-download concurrency for ~2.5x throughput.
    # (No-op-safe; for large files you'd skip this. See bucketbag.boost docstring.)
    boost(file_concurrency=32)

    create_bucket(OUT, exist_ok=True)

    # Resume: list candidate keys, drop the ones already written.
    all_keys = list(iter_keys(SRC, include=args.include, limit=args.limit))
    done = completed_keys(OUT)
    todo = [k for k in all_keys if k not in done]
    skipped = len(all_keys) - len(todo)
    print(
        f"{len(all_keys)} candidate pages | {skipped} already done (skipped) | {len(todo)} to do",
        flush=True,
    )
    if not todo:
        print("Nothing to do — resume complete.", flush=True)
        return

    t0 = time.monotonic()
    high_water = 0.0
    processed = 0
    for batch in batched_files(SRC, keys=todo, n=args.n, prefetch=args.prefetch, dir=scratch):
        high_water = max(high_water, dir_size_mb(scratch))
        rows = [{"__source_key": it.key, "n_bytes": len(it.bytes)} for it in batch]
        # deterministic, idempotent shard name from the first key in the batch
        name = "part-" + batch[0].key.replace("/", "_").rsplit(".", 1)[0] + ".parquet"
        write_parquet(rows, OUT, name)
        processed += len(batch)
        print(
            f"  +{len(batch):>3} pages (total {processed}/{len(todo)}) "
            f"scratch high-water {high_water:.0f} MB",
            flush=True,
        )

    wall = time.monotonic() - t0
    print(
        f"\nDONE: {processed} pages in {wall:.1f}s = {processed / wall:.1f} pages/s\n"
        f"peak RSS {peak_rss_mb():.0f} MB | scratch high-water {high_water:.0f} MB "
        f"(bounded to ~{args.prefetch + 1} batches of {args.n})\n"
        f"Re-run to verify resume skips everything.",
        flush=True,
    )


if __name__ == "__main__":
    main()
