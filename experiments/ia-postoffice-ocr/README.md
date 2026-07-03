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

## Results

*(filled in as runs complete)*

| run | flavor | pages | batch | prefetch | pages/min | inference share of wall | $/1k pages |
| --- | --- | --- | --- | --- | --- | --- | --- |

Notes / gotchas:

- (pending)
