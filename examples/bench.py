# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bucketbag @ git+https://github.com/davanstrien/bucketbag",
# ]
# ///
"""Benchmark bucketbag's read path against the alternatives, on the SAME fixed set of files.

Methods (pick with --methods):
  - bucketbag : `batched_files` (download+cleanup loop); swept over --prefetch values
  - download  : raw `download_bucket_files` batched to a tmp dir (no prefetch) — the baseline
  - mount     : read from a FUSE-mounted bucket path (--mount); only exists inside a Job
  - fsspec    : `HfFileSystem` signed-URL range reads, threaded (the faster-but-cautioned path)

Reports imgs/s, MB/s, peak RSS, and scratch high-water (proves bounded disk).

Local sanity (small --n, no mount):
  uv run examples/bench.py --n 120 --methods bucketbag,download,fsspec --prefetch 0,1,2

Realistic, on a Job with the bucket mounted (mount only exists there):
  hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN=$HF_TOKEN -e HF_XET_HIGH_PERFORMANCE=1 \\
    -v hf://buckets/davanstrien/finebooks-bhl-pilot:/bucket examples/bench.py -- \\
    --n 3000 --methods mount,download,bucketbag,fsspec --prefetch 0,2
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1024


def dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1e6


def _safe(fn, k):
    try:
        return fn(k)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="davanstrien/finebooks-bhl-pilot")
    ap.add_argument("--include", default="images/**/*.jp2")
    ap.add_argument("--prefix", default="images")
    ap.add_argument("--n", type=int, default=120, help="number of files to benchmark")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=32, help="threads for mount/fsspec")
    ap.add_argument("--prefetch", default="0,2", help="csv of prefetch values to sweep (bucketbag)")
    ap.add_argument("--work-ms", type=int, default=0, help="simulated per-batch compute (ms)")
    ap.add_argument("--mount", default=None, help="FUSE mount path of the bucket (Job only)")
    ap.add_argument("--methods", default="bucketbag,download,fsspec")
    args = ap.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    import bucketbag as bb

    api = HfApi()

    # Enumerate a fixed set once: keys + BucketFile objects, reused by every method.
    t = time.monotonic()
    keys: list[str] = []
    bfs = []
    for f in api.list_bucket_tree(args.bucket, args.prefix, recursive=True):
        p = getattr(f, "path", None)
        if p and getattr(f, "type", None) == "file" and p.lower().endswith(".jp2"):
            keys.append(p)
            bfs.append(f)
            if len(keys) >= args.n:
                break
    n = len(keys)
    print(f"ENUM n={n} in {time.monotonic() - t:.1f}s\n", flush=True)

    scratch = Path(os.environ.get("BUCKETBAG_DIR", "/dev/shm"))
    if not (scratch.is_dir() and os.access(scratch, os.W_OK)):
        scratch = Path(tempfile.gettempdir())

    results: list[str] = []

    def line(name, secs, total_bytes, fails, extra=""):
        ok = n - fails
        imgs_s = ok / max(secs, 0.1)
        mb_s = total_bytes / 1e6 / max(secs, 0.1)
        results.append(
            f"{name:<28} imgs/s={imgs_s:6.1f}  MB/s={mb_s:6.1f}  ok={ok} fails={fails}  {extra}"
        )
        print(results[-1], flush=True)

    methods = args.methods.split(",")

    if "bucketbag" in methods:
        for p in (int(x) for x in args.prefetch.split(",")):
            run_dir = scratch / "bb-bench"
            shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            t = time.monotonic()
            tot = fails = 0
            high = 0.0
            for batch in bb.batched_files(
                args.bucket, keys=bfs, n=args.batch_size, prefetch=p, dir=run_dir
            ):
                high = max(high, dir_size_mb(run_dir))
                for it in batch:
                    tot += len(it.bytes)
                fails += args.batch_size - len(batch) if len(batch) < args.batch_size else 0
                if args.work_ms:
                    time.sleep(args.work_ms / 1000.0)
            line(
                f"bucketbag(prefetch={p},n={args.batch_size})",
                time.monotonic() - t,
                tot,
                0,
                f"scratch high-water {high:.0f} MB  peak RSS {peak_rss_mb():.0f} MB",
            )
            shutil.rmtree(run_dir, ignore_errors=True)

    if "download" in methods:
        t = time.monotonic()
        tot = fails = 0
        peak = 0.0
        for i in range(0, n, args.batch_size):
            tmp = Path(tempfile.mkdtemp(dir=scratch))
            try:
                end = min(i + args.batch_size, n)
                pairs = [(bfs[j], tmp / f"{j:07d}.jp2") for j in range(i, end)]
                api.download_bucket_files(args.bucket, files=pairs)
                here = 0
                for _, lp in pairs:
                    if Path(lp).exists():
                        here += Path(lp).stat().st_size
                    else:
                        fails += 1
                tot += here
                peak = max(peak, here / 1e6)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        line(
            f"download(batch={args.batch_size})",
            time.monotonic() - t,
            tot,
            fails,
            f"peak per-batch {peak:.0f} MB",
        )

    if "mount" in methods:
        if not args.mount:
            print("mount: skipped (no --mount path; mount only exists inside a Job)", flush=True)
        else:

            def rd_mount(k):
                with open(os.path.join(args.mount, k), "rb") as fh:
                    return len(fh.read())

            t = time.monotonic()
            tot = fails = 0
            with cf.ThreadPoolExecutor(args.workers) as ex:
                for r in ex.map(lambda k: _safe(rd_mount, k), keys):
                    tot += r or 0
                    fails += 0 if r else 1
            line(f"mount(threads={args.workers})", time.monotonic() - t, tot, fails)

    if "fsspec" in methods:
        fs = HfFileSystem()

        def rd_fs(k):
            with fs.open(f"hf://buckets/{args.bucket}/{k}", "rb") as fh:
                return len(fh.read())

        t = time.monotonic()
        tot = fails = 0
        with cf.ThreadPoolExecutor(args.workers) as ex:
            for r in ex.map(lambda k: _safe(rd_fs, k), keys):
                tot += r or 0
                fails += 0 if r else 1
        line(f"fsspec(threads={args.workers})", time.monotonic() - t, tot, fails)

    print("\n=== summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
