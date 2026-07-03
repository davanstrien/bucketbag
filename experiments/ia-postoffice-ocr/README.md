# Experiment: Surya OCR 2 over IA post-office directories, driven by `Bag`

First real-Jobs dogfood of the `Bag` interface: OCR historical Glasgow Post Office
directories (dense multi-column pages — a layout-heavy workload) straight from a bucket,
measuring end-to-end throughput of the list → resume-skip → batched-download → OCR →
`put_files` loop.

## Data

Three consecutive Internet Archive items (Glasgow PO directories, public domain,
~2,280 pages each, ~780 KB/page jp2):

- [`postofficean192829glas`](https://archive.org/details/postofficean192829glas)
- [`postofficean192930glas`](https://archive.org/details/postofficean192930glas)
- [`postofficean193031glas`](https://archive.org/details/postofficean193031glas)

Mirrored to `hf://buckets/davanstrien/ia-postoffice-directories/{item_id}/...` by
`mirror_ia.py` (CPU Job: stream the `_jp2.zip`, extract, upload via `put_files`,
resume-safe):

```bash
hf jobs uv run --flavor cpu-upgrade --timeout 2h -s HF_TOKEN \
    experiments/ia-postoffice-ocr/mirror_ia.py
```

## OCR

`ocr_bag.py` = a `Bag` loop + the offline-vLLM Surya engine vendored **unchanged** from
uv-scripts-for-ai's `ocr/surya-ocr-bucket.py` (model `datalab-to/surya-ocr-2`, 650M).
Outputs per page under `<src>/surya/`: `<key>.md` (reading-order text) + `<key>.json`
(structured blocks + boxes + confidence).

```bash
# smoke
hf jobs uv run --flavor l4x1 --timeout 1h -s HF_TOKEN \
    --image vllm/vllm-openai:v0.20.1 --python /usr/local/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/site-packages \
    experiments/ia-postoffice-ocr/ocr_bag.py --limit 24

# throughput run
hf jobs uv run --flavor l4x1 --timeout 2h -s HF_TOKEN \
    --image vllm/vllm-openai:v0.20.1 --python /usr/local/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/site-packages \
    experiments/ia-postoffice-ocr/ocr_bag.py --limit 500
```

The whole I/O side of the script is:

```python
(Bag.from_bucket(src, include="**/*.jp2", limit=args.limit)
   .map_batches(run, batch_size=16, setup=load_engine)
   .to_bucket(f"{src}/surya")
   .compute(prefetch=2))
```

Resume = re-run the same command; pages with outputs are skipped.

## Results (2026-07-03)

| run | flavor | pages | pages/min | inference share of wall | $/1k pages |
| --- | --- | --- | --- | --- | --- |
| mirror (IA → bucket, cpu) | cpu-upgrade | 6,904 | ~2,000 (34/s upload) | n/a | ~$0.03 |
| Bag smoke (batch 16, prefetch 2) | a10g-small | 24 | 3.3 (incl. engine load) | — | — |
| Bag 500-page run | a10g-small | 476 | **7.7** (8.0 steady-state) | **~88%** | **~$2.2** |
| recipe `--io-mode mount` (same pages) | a10g-small | 500 | 7.5 | ~86% (serial FUSE reads = 13.6%) | ~$2.2 |

- **Resume verified on real state**: the 500-run logged `24 inputs already done, 476 to do` and skipped the smoke's pages. Listing + resume-scan overhead: **~3 s** of a 62-min job.
- **The workload is inference-bound (~88%)**, so the I/O path is invisible in throughput terms —
  Surya at ~0.13 pages/s is far below even a slow read path. bucketbag's contribution here is
  not speed: it's the ~6-line I/O layer, bounded scratch, and **resume across the timeout
  ceiling** (the full 6,900-page corpus is ~14 h on one GPU — only runnable as kill/re-run,
  or fanned out across jobs, which is the planned `Bag` next layer).
- Cost: ~$0.0022/page with a 650M layout-aware model on $1/hr hardware.
- **Mount comparison (unmodified upstream recipe, same 500 pages, same flavor):** 8.0 s/page
  vs Bag's 7.8 s/page — a ~3% difference, i.e. **the I/O path is irrelevant for VLM-class OCR**,
  as the inference share predicted. Engine speed identical (6.9 s/page both). The measured
  difference in kind: mount reads are **serial** at 0.92 files/s (545 s of the loop); Bag's
  prefetch overlaps downloads with compute. With a model ~10× faster than Surya (PP-OCR-class),
  serial 1.1 s/page FUSE reads would cap throughput at ~55 pages/min while the overlapped API
  path keeps the GPU fed — that's the crossover where the read path starts to matter.
- **Output parity: mean text similarity 1.000** across a 12-page stratified sample
  (500/500 outputs present on both sides; two pages differ by 1–2 chars — token-margin
  nondeterminism). The Bag port reproduces the recipe exactly. FUSE writes also all landed
  in this run.

Gotchas hit (each one commit):

- `surya-ocr==0.20.0` pins `huggingface-hub<1`; bucketbag needs `>=1.12` — unsolvable resolve,
  escaped via `[tool.uv] override-dependencies` (safe here: the image's hub 1.x wins on
  PYTHONPATH at runtime, the same runtime the upstream recipe is validated on).
- The vllm-openai image has **no git** — install bucketbag from the GitHub **tag tarball**,
  not `git+`.
- l4x1 sat in SCHEDULING; a10g-small scheduled immediately (same 24 GB class).
- Bucket listing prefixes are **string** prefixes: `prefix="surya"` also matches `surya-mount/...`.
  Use a trailing-slash-aware filter when prefixes can collide.
- Key conventions differ: the recipe *replaces* the extension (`X.md`), bucketbag examples
  *append* (`X.jp2.md` — which the Bag resume contract needs, since done-ness is
  output-key-starts-with-source-key).
