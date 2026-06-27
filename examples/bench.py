# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bucketbag @ git+https://github.com/davanstrien/bucketbag",
# ]
# ///
"""Benchmark bucketbag's read path against the alternatives, on the SAME fixed set of files.

Methods (pick with --methods):
  - bucketbag : `batched_files` swept over a prefetch x batch-size GRID (find the optimum)
  - download  : raw `download_bucket_files` batched to a tmp dir (no prefetch) — the baseline
  - mount     : read from a FUSE-mounted bucket path (--mount); only exists inside a Job
  - fsspec    : `HfFileSystem` signed-URL range reads, threaded (the faster-but-cautioned path)

Reports imgs/s, MB/s, peak RSS, scratch high-water, and the BEST bucketbag config.

Local sanity (small --n):
  uv run examples/bench.py --n 120 --prefetch 0,2 --bb-batch-sizes 32,64

Grid on a Job (mount only exists there):
  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e HF_XET_HIGH_PERFORMANCE=1 \\
    -v hf://buckets/davanstrien/finebooks-bhl-pilot:/bucket:ro examples/bench.py -- \\
    --n 2000 --prefetch 0,1,2,4,8 --bb-batch-sizes 32,64,128 \\
    --methods bucketbag,download,mount,fsspec --mount /bucket
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
    ap.add_argument("--prefix", default="images")
    ap.add_argument("--n", type=int, default=120, help="number of files to benchmark")
    ap.add_argument("--batch-size", type=int, default=64, help="batch size for download baseline")
    ap.add_argument("--bb-batch-sizes", default="", help="csv batch sizes for bucketbag grid")
    ap.add_argument("--prefetch", default="0,2", help="csv prefetch values (bucketbag grid)")
    ap.add_argument("--workers", type=int, default=32, help="threads for mount/fsspec")
    ap.add_argument("--work-ms", type=int, default=0, help="simulated per-batch compute (ms)")
    ap.add_argument("--mount", default=None, help="FUSE mount path of the bucket (Job only)")
    ap.add_argument("--methods", default="bucketbag,download,fsspec")
    args = ap.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    import bucketbag as bb

    # node info — so each flavor's results are self-identifying
    shm = Path("/dev/shm")
    shm_gb = (shutil.disk_usage(shm).total / 1e9) if shm.is_dir() else 0.0
    ram_gb = 0.0
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    print(
        f"NODE cpus={os.cpu_count()} ram={ram_gb:.0f}GB /dev/shm={shm_gb:.0f}GB "
        f"xet_hp={os.environ.get('HF_XET_HIGH_PERFORMANCE', '0')}",
        flush=True,
    )

    api = HfApi()
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
    total_mb = sum(getattr(f, "size", 0) for f in bfs) / 1e6
    avg_mb = total_mb / max(n, 1)
    print(
        f"ENUM n={n} ({total_mb:.0f} MB, {avg_mb:.2f} MB/file) in {time.monotonic() - t:.1f}s\n",
        flush=True,
    )

    scratch = Path(os.environ.get("BUCKETBAG_DIR", "/dev/shm"))
    if not (scratch.is_dir() and os.access(scratch, os.W_OK)):
        scratch = Path(tempfile.gettempdir())

    results: list[str] = []
    metrics: list[tuple[str, float]] = []

    def line(name, secs, total_bytes, fails, extra=""):
        ok = n - fails
        imgs_s = ok / max(secs, 0.1)
        mb_s = total_bytes / 1e6 / max(secs, 0.1)
        metrics.append((name, imgs_s))
        results.append(
            f"{name:<30} imgs/s={imgs_s:6.1f}  MB/s={mb_s:6.1f}  ok={ok} fails={fails}  {extra}"
        )
        print(results[-1], flush=True)

    methods = args.methods.split(",")

    if "bucketbag" in methods:
        pfs = [int(x) for x in args.prefetch.split(",")]
        bss = [int(x) for x in (args.bb_batch_sizes or str(args.batch_size)).split(",")]
        for bs in bss:
            for p in pfs:
                run_dir = scratch / "bb-bench"
                shutil.rmtree(run_dir, ignore_errors=True)
                run_dir.mkdir(parents=True, exist_ok=True)
                t = time.monotonic()
                tot = got = 0
                high = 0.0
                for batch in bb.batched_files(args.bucket, keys=bfs, n=bs, prefetch=p, dir=run_dir):
                    high = max(high, dir_size_mb(run_dir))
                    for it in batch:
                        tot += len(it.bytes)
                        got += 1
                    if args.work_ms:
                        time.sleep(args.work_ms / 1000.0)
                line(
                    f"bucketbag(prefetch={p},n={bs})",
                    time.monotonic() - t,
                    tot,
                    n - got,
                    f"high-water {high:.0f} MB  RSS {peak_rss_mb():.0f} MB",
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
            "no read (stat only)",
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

    print("\n=== summary (sorted by imgs/s) ===")
    for r in sorted(results, key=lambda s: float(s.split("imgs/s=")[1].split()[0]), reverse=True):
        print(r)
    bb_metrics = [(name, v) for name, v in metrics if name.startswith("bucketbag")]
    if bb_metrics:
        best = max(bb_metrics, key=lambda x: x[1])
        print(f"\nBEST bucketbag config: {best[0]} @ {best[1]:.1f} imgs/s", flush=True)


if __name__ == "__main__":
    main()
