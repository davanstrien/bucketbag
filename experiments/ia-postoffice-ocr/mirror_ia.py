# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bucketbag @ git+https://github.com/davanstrien/bucketbag@v0.2.0",
#     "requests",
# ]
# ///
"""Mirror Internet Archive post-office-directory page images into an HF bucket.

For each IA item: stream-download `{id}_jp2.zip`, extract, upload the `.jp2` pages to the
bucket via ``put_files`` (batched), then delete local files. Resumable: pages already in the
bucket are skipped, and a partially-mirrored item picks up where it left off.

Run on HF Jobs (CPU):

    hf jobs uv run --flavor cpu-upgrade --timeout 2h -s HF_TOKEN \\
        experiments/ia-postoffice-ocr/mirror_ia.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from huggingface_hub import create_bucket

from bucketbag import iter_keys, partition_all, put_files

DEFAULT_ITEMS = [
    "postofficean192829glas",
    "postofficean192930glas",
    "postofficean193031glas",
]
DEFAULT_BUCKET = "davanstrien/ia-postoffice-directories"
UPLOAD_CHUNK = 64


def mirror_item(item_id: str, bucket: str, limit: int | None) -> None:
    done = set(iter_keys(bucket, prefix=item_id))
    url = f"https://archive.org/download/{item_id}/{item_id}_jp2.zip"
    workdir = Path(tempfile.mkdtemp(prefix=f"ia-{item_id}-"))
    try:
        zpath = workdir / "pages.zip"
        t0 = time.monotonic()
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zpath, "wb") as fh:
                shutil.copyfileobj(r.raw, fh, length=1 << 20)
        dl_s = time.monotonic() - t0
        gb = zpath.stat().st_size / 1e9
        print(f"{item_id}: downloaded {gb:.2f} GB in {dl_s:.0f}s", flush=True)

        with zipfile.ZipFile(zpath) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(".jp2"))
            if limit:
                names = names[:limit]
            todo = [n for n in names if f"{item_id}/{n}" not in done]
            mirrored = len(names) - len(todo)
            print(f"{item_id}: {len(names)} pages, {mirrored} already mirrored", flush=True)
            t0 = time.monotonic()
            sent = 0
            for chunk in partition_all(UPLOAD_CHUNK, todo):
                pairs = [(f"{item_id}/{name}", zf.read(name)) for name in chunk]
                put_files(pairs, bucket)
                sent += len(pairs)
                print(f"{item_id}: uploaded {sent}/{len(todo)}", flush=True)
        if todo:
            up_s = time.monotonic() - t0
            print(f"{item_id}: upload {up_s:.0f}s = {sent / up_s:.1f} pages/s", flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", nargs="*", default=DEFAULT_ITEMS, help="IA identifiers")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--limit", type=int, default=None, help="max pages per item (smoke runs)")
    args = ap.parse_args()

    create_bucket(args.bucket, exist_ok=True)
    for item_id in args.items:
        mirror_item(item_id, args.bucket, args.limit)
    print("done", flush=True)


if __name__ == "__main__":
    main()
