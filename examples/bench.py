# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bucketbag @ git+https://github.com/davanstrien/bucketbag",
# ]
# ///
"""Benchmark bucketbag's read path — methodologically careful edition.

Answers: which (prefetch, batch) config is best, and does `batched_files` beat the
alternatives (raw download_bucket_files, FUSE mount, HfFileSystem)?

Key design choices (so the numbers are trustworthy):
  * COLD + DISJOINT: every (config, replicate) reads its OWN non-overlapping, shuffled
    set of files. This matches the real workload (each file read once) AND neutralizes the
    xet local chunk cache — otherwise re-reading the same files makes later runs look fast.
  * REPLICATED + RANDOMIZED ORDER: each config runs --replicates times in shuffled order;
    we report the MEDIAN so one noisy run or any residual warm-up can't crown a winner.
  * FAIR: every method READS the file bytes (the download baseline too), uses BucketFile
    objects (no metadata HEAD), and runs on its own cold slice.
  * Disk high-water is sampled in a BACKGROUND thread (never inside the timed loop).
  * Per-run failures are isolated; a headroom guard skips configs that wouldn't fit on scratch.

Local sanity (small):
  uv run examples/bench.py --n 60 --replicates 1 --prefetch 0,2 --bb-batch-sizes 32,64

On a Job (mount only exists there):
  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e HF_XET_HIGH_PERFORMANCE=1 \\
    -v hf://buckets/davanstrien/finebooks-bhl-pilot:/bucket:ro examples/bench.py -- \\
    --n 300 --replicates 3 --prefetch 0,1,2,4,8 --bb-batch-sizes 32,64,128 \\
    --methods bucketbag,download,mount,fsspec --mount /bucket
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import functools
import os
import random
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from statistics import median, pstdev


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1024


class DiskHighWater(threading.Thread):
    """Sample free/used bytes on a path in the background; report MB above baseline."""

    def __init__(self, path: Path, interval: float = 0.05):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self._stopev = threading.Event()  # NOT _stop: that shadows Thread._stop()
        self.base = shutil.disk_usage(path).used
        self.peak = self.base

    def run(self) -> None:
        while not self._stopev.is_set():
            try:
                self.peak = max(self.peak, shutil.disk_usage(self.path).used)
            except OSError:
                pass
            self._stopev.wait(self.interval)

    def stop_mb(self) -> float:
        self._stopev.set()
        self.join(timeout=1)
        return max(0.0, (self.peak - self.base) / 1e6)


def _safe(fn, k):
    try:
        return fn(k)
    except Exception:
        return None


def _read_path(path: str) -> int:
    with open(path, "rb") as fh:
        return len(fh.read())


def _read_fsspec(fs, url: str) -> int:
    with fs.open(url, "rb") as fh:
        return len(fh.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="davanstrien/finebooks-bhl-pilot")
    ap.add_argument("--prefix", default="images")
    ap.add_argument("--n", type=int, default=300, help="files per run (per config, per replicate)")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--prefetch", default="0,1,2,4,8", help="csv prefetch values (bucketbag grid)")
    ap.add_argument(
        "--bb-batch-sizes", default="32,64,128", help="csv batch sizes (bucketbag grid)"
    )
    ap.add_argument("--workers", type=int, default=32, help="threads for mount/fsspec")
    ap.add_argument("--work-ms", type=int, default=0, help="simulated per-batch compute (ms)")
    ap.add_argument("--mount", default=None, help="FUSE mount path of the bucket (Job only)")
    ap.add_argument("--methods", default="bucketbag,download,fsspec")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    import bucketbag as bb

    rng = random.Random(args.seed)
    methods = args.methods.split(",")
    prefetches = [int(x) for x in args.prefetch.split(",")]
    batch_sizes = [int(x) for x in args.bb_batch_sizes.split(",")]

    shm = Path("/dev/shm")
    shm_gb = (shutil.disk_usage(shm).total / 1e9) if shm.is_dir() else 0.0
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        ram_gb = 0.0
    print(
        f"NODE cpus={os.cpu_count()} ram={ram_gb:.0f}GB /dev/shm={shm_gb:.0f}GB "
        f"xet_hp={os.environ.get('HF_XET_HIGH_PERFORMANCE', '0')}",
        flush=True,
    )

    scratch = Path(os.environ.get("BUCKETBAG_DIR", "/dev/shm"))
    if not (scratch.is_dir() and os.access(scratch, os.W_OK)):
        scratch = Path(tempfile.gettempdir())

    # ---- build the run plan (one disjoint cold slice per run) ----
    runs: list[tuple[str, tuple | None]] = []
    if "bucketbag" in methods:
        for bs in batch_sizes:
            for p in prefetches:
                runs.append(("bucketbag", (p, bs)))
    for m in ("download", "mount", "fsspec"):
        if m in methods:
            runs.append((m, None))
    runs = runs * args.replicates
    rng.shuffle(runs)

    warmup = 8
    need_keys = len(runs) * args.n + warmup

    api = HfApi()
    t = time.monotonic()
    bfs = []
    for f in api.list_bucket_tree(args.bucket, args.prefix, recursive=True):
        p = getattr(f, "path", None)
        if p and getattr(f, "type", None) == "file" and p.lower().endswith(".jp2"):
            bfs.append(f)
            if len(bfs) >= need_keys:
                break
    rng.shuffle(bfs)
    avg_mb = (sum(getattr(f, "size", 0) for f in bfs) / max(len(bfs), 1)) / 1e6
    print(
        f"ENUM {len(bfs)} keys (need {need_keys}, avg {avg_mb:.2f} MB/file) in "
        f"{time.monotonic() - t:.1f}s | {len(runs)} runs ({args.replicates}x), cold+disjoint\n",
        flush=True,
    )
    if len(bfs) < need_keys:
        print(f"WARNING: only {len(bfs)} keys available; slices will be smaller / may overlap")

    # warm up connection/TLS/auth so the first timed run isn't penalized
    warm_dir = Path(tempfile.mkdtemp(dir=scratch))
    try:
        api.download_bucket_files(
            args.bucket,
            files=[(bfs[-i - 1], warm_dir / f"w{i}") for i in range(min(warmup, len(bfs)))],
        )
    except Exception:
        pass
    finally:
        shutil.rmtree(warm_dir, ignore_errors=True)

    fs = HfFileSystem() if "fsspec" in methods else None
    samples: dict[str, list[float]] = {}
    hw_by: dict[str, list[float]] = {}

    for idx, (kind, params) in enumerate(runs):
        sl = bfs[idx * args.n : (idx + 1) * args.n]
        if not sl:
            continue
        keys = [f.path for f in sl]
        try:
            if kind == "bucketbag":
                p, bs = params
                need_mb = (p + 1) * bs * max(avg_mb, 0.1)
                if shutil.disk_usage(scratch).free / 1e6 < need_mb * 1.5:
                    print(
                        f"  skip bucketbag(p={p},n={bs}): not enough scratch headroom", flush=True
                    )
                    continue
                run_dir = Path(tempfile.mkdtemp(dir=scratch, prefix="bbx-"))
                hw = DiskHighWater(scratch)
                hw.start()
                t = time.monotonic()
                got = 0
                for batch in bb.batched_files(args.bucket, keys=sl, n=bs, prefetch=p, dir=run_dir):
                    for it in batch:
                        got += len(it.bytes)  # READ the bytes (fair)
                    if args.work_ms:
                        time.sleep(args.work_ms / 1000.0)
                secs = time.monotonic() - t
                label = f"bucketbag(prefetch={p},n={bs})"
                hw_by.setdefault(label, []).append(hw.stop_mb())
                shutil.rmtree(run_dir, ignore_errors=True)

            elif kind == "download":
                run_dir = Path(tempfile.mkdtemp(dir=scratch, prefix="dlx-"))
                hw = DiskHighWater(scratch)
                hw.start()
                t = time.monotonic()
                pairs = [(sl[j], run_dir / f"{j:06d}.jp2") for j in range(len(sl))]
                api.download_bucket_files(args.bucket, files=pairs)
                got = sum(Path(lp).read_bytes().__len__() for _, lp in pairs if Path(lp).exists())
                secs = time.monotonic() - t
                label = "download(read,n=all)"
                hw_by.setdefault(label, []).append(hw.stop_mb())
                shutil.rmtree(run_dir, ignore_errors=True)

            elif kind == "mount":
                if not args.mount:
                    continue
                paths = [os.path.join(args.mount, k) for k in keys]
                reader = functools.partial(_safe, _read_path)
                t = time.monotonic()
                with cf.ThreadPoolExecutor(args.workers) as ex:
                    got = sum(r or 0 for r in ex.map(reader, paths))
                secs = time.monotonic() - t
                label = f"mount(threads={args.workers})"

            elif kind == "fsspec":
                urls = [f"hf://buckets/{args.bucket}/{k}" for k in keys]
                reader = functools.partial(_safe, functools.partial(_read_fsspec, fs))
                t = time.monotonic()
                with cf.ThreadPoolExecutor(args.workers) as ex:
                    got = sum(r or 0 for r in ex.map(reader, urls))
                secs = time.monotonic() - t
                label = f"fsspec(threads={args.workers})"
            else:
                continue

            imgs_s = len(sl) / max(secs, 1e-6)
            samples.setdefault(label, []).append(imgs_s)
            print(f"  [{idx + 1}/{len(runs)}] {label:<28} {imgs_s:6.1f} img/s", flush=True)
        except Exception as e:  # noqa: BLE001 - isolate one run's failure
            print(f"  [{idx + 1}/{len(runs)}] {kind} {params} FAILED: {e}", flush=True)

    # ---- aggregate ----
    print("\n=== results (median over replicates, sorted) ===", flush=True)
    rows = []
    for label, vals in samples.items():
        med = median(vals)
        spread = pstdev(vals) if len(vals) > 1 else 0.0
        hw = max(hw_by.get(label, [0.0]))
        rows.append((med, label, spread, len(vals), hw))
    for med, label, spread, nrep, hw in sorted(rows, reverse=True):
        hw_s = f"  high-water {hw:.0f} MB" if hw else ""
        print(f"{label:<30} {med:6.1f} img/s  (±{spread:4.1f}, n={nrep}){hw_s}", flush=True)

    bb_rows = [r for r in rows if r[1].startswith("bucketbag")]
    if bb_rows:
        best = max(bb_rows, key=lambda r: r[0])
        print(f"\nBEST bucketbag: {best[1]} @ {best[0]:.1f} img/s (±{best[2]:.1f})", flush=True)
    print(f"peak RSS {peak_rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
