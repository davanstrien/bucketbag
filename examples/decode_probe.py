# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bucketbag @ git+https://github.com/davanstrien/bucketbag@v0.3.1",
#     "pillow",
# ]
# ///
"""Is it the transport or the decode? Time the two separately.

Pass 1 downloads a slice of jp2 pages with ``batched_files`` and only stats them. Pass 2 downloads
the same slice again and decodes each page with Pillow. Pass 3 decodes one batch at each JPEG 2000
``reduce`` level. Run it before reaching for a faster transport:

    uv run examples/decode_probe.py                         # 48 pages, 3 passes
    hf jobs uv run --flavor cpu-basic --timeout 15m examples/decode_probe.py -- --limit 96

Findings on BHL pages (~4300x5500 px, ~0.7 MB, cpu-basic): download 40 files in ~1 s; full-res
Pillow decode ~1.7 s/page and it holds the GIL (threads don't help; processes do); ``reduce=3``
decodes in ~0.06 s. See README "Decode, not transport".
"""

from __future__ import annotations

import argparse
import math
import time

from PIL import Image

from bucketbag import batched_files, iter_keys

SRC = "davanstrien/finebooks-bhl-pilot"


def safe_reduce(width: int, height: int, *, target: int = 512, max_level: int = 3) -> int:
    """Largest JPEG 2000 ``reduce`` level that keeps the short side >= ``target`` and that Pillow
    can actually load: Pillow sizes the reduced image as ``int((d + p/2) / p)`` while OpenJPEG uses
    ``ceil(d / p)``; when they disagree for either dimension ``load()`` fails with
    "broken data stream". 0 means full-resolution decode.
    """
    for level in range(max_level, 0, -1):
        power = 1 << level
        adjust = power >> 1
        if min(width, height) // power < target:
            continue
        if all(int((d + adjust) / power) == math.ceil(d / power) for d in (width, height)):
            return level
    return 0


def decode(path, reduce: int = 0) -> Image.Image:
    im = Image.open(path)
    if reduce:
        im.reduce = reduce  # must be set before load()
    return im.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=48, help="number of pages")
    ap.add_argument("--n", type=int, default=16, help="files per batch")
    ap.add_argument("--include", default="images/**/*.jp2", help="glob to select pages")
    args = ap.parse_args()

    keys = list(iter_keys(SRC, include=args.include, limit=args.limit))

    # Pass 1: transport only.
    t0 = time.monotonic()
    n = nbytes = 0
    for batch in batched_files(SRC, keys=keys, n=args.n, prefetch=2):
        n += len(batch)
        nbytes += sum(it.path.stat().st_size for it in batch)
    t_dl = time.monotonic() - t0
    print(f"download: {n} files, {nbytes / 1e6:.1f} MB in {t_dl:.1f}s -> {n / t_dl:.1f} img/s")

    # Pass 2: transport + full-resolution decode (what a naive loop pays).
    t0 = time.monotonic()
    size = None
    for batch in batched_files(SRC, keys=keys, n=args.n, prefetch=2):
        for it in batch:
            size = Image.open(it.path).size
            decode(it.path)
    t_all = time.monotonic() - t0
    print(
        f"download+decode: {n} files in {t_all:.1f}s -> {n / t_all:.1f} img/s "
        f"(decode ~{(t_all - t_dl) / n:.2f} s/img at {size})"
    )

    # Pass 3: one batch at each reduce level. Levels the rounding rule rejects are skipped.
    # (Stay inside the loop: the batch's temp dir is removed when iteration moves on.)
    for batch in batched_files(SRC, keys=keys, n=args.n, prefetch=0):
        for level in range(4):
            timed = 0
            t0 = time.monotonic()
            for it in batch:
                w, h = Image.open(it.path).size
                if level and safe_reduce(w, h, target=1, max_level=level) != level:
                    continue
                decode(it.path, level)
                timed += 1
            dt = time.monotonic() - t0
            if timed:
                print(f"reduce={level}: {dt / timed:.3f} s/img ({timed}/{len(batch)} loadable)")
        break


if __name__ == "__main__":
    main()
