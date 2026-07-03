# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "bucketbag @ git+https://github.com/davanstrien/bucketbag@v0.2.0",
#     "surya-ocr==0.20.0",
#     "beautifulsoup4",
#     "imagecodecs",
#     "pillow",
# ]
#
# # surya-ocr pinned to the known-good build with the `surya.inference` engine layout
# # (same pin rationale as uv-scripts-for-ai's surya recipes). vLLM and torch come from
# # the vllm-openai IMAGE via PYTHONPATH, not this venv.
# ///
"""bucketbag throughput experiment: Surya OCR 2 over IA post-office directories.

The I/O loop is bucketbag's ``Bag`` (list -> resume-skip -> batched download -> OCR ->
``put_files``); the engine glue (offline-vLLM backend wired into Surya's predictor stack)
is vendored unchanged from uv-scripts-for-ai ``ocr/surya-ocr-bucket.py`` / ``surya-ocr.py``.
Outputs per page: ``<key>.md`` (reading-order text) + ``<key>.json`` (structured blocks).

Run on HF Jobs (same image/PYTHONPATH pattern as the recipe):

    hf jobs uv run --flavor l4x1 --timeout 2h -s HF_TOKEN \\
        --image vllm/vllm-openai:v0.20.1 --python /usr/local/bin/python3 \\
        -e PYTHONPATH=/usr/local/lib/python3.12/site-packages \\
        experiments/ia-postoffice-ocr/ocr_bag.py --limit 24        # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from bucketbag import Bag, LoadedItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr-bag")

DEFAULT_SRC = "davanstrien/ia-postoffice-directories"
DEFAULT_MODEL = "datalab-to/surya-ocr-2"
MM_PROCESSOR_KWARGS = {"min_pixels": 3136, "max_pixels": 6291456}
JP2_EXTENSIONS = {".jp2", ".j2k"}


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA required — run on a GPU Job (see module docstring).")
        sys.exit(1)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def open_image(path: Path) -> Image.Image:
    """Open one image as RGB; imagecodecs fallback for JPEG-2000 without OpenJPEG.
    (Vendored from surya-ocr-bucket.py.)"""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        if path.suffix.lower() in JP2_EXTENSIONS:
            import imagecodecs

            arr = imagecodecs.imread(str(path))
            return Image.fromarray(arr).convert("RGB")
        raise


# ---------------------------------------------------------------------------
# Offline vLLM backend + Surya manager — vendored unchanged from
# uv-scripts-for-ai ocr/surya-ocr-bucket.py (itself verbatim from surya-ocr.py).
# ---------------------------------------------------------------------------


def build_structured_outputs(schema: dict[str, Any]) -> dict[str, Any]:
    """SamplingParams kwargs for guided JSON, across vLLM versions."""
    try:
        from vllm.sampling_params import StructuredOutputsParams  # vLLM >= 0.12

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except (ImportError, TypeError):
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams  # older vLLM

        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    except (ImportError, TypeError):
        pass
    logger.warning("Guided JSON unavailable in this vLLM version; relying on the model.")
    return {}


def _mean_token_prob(completion_output) -> float | None:
    """Mean exp(logprob) of the sampled tokens -> Surya's per-block `confidence`."""
    lps = getattr(completion_output, "logprobs", None)
    if not lps:
        return None
    probs: list[float] = []
    for tid, lp_dict in zip(completion_output.token_ids, lps):
        if not lp_dict:
            continue
        entry = lp_dict.get(tid)
        if entry is None:  # sampled token not in the returned top-k; use the best we have
            entry = max(lp_dict.values(), key=lambda e: e.logprob)
        probs.append(math.exp(entry.logprob))
    return sum(probs) / len(probs) if probs else None


class OfflineVLLMBackend:
    """Surya `Backend` (duck-typed) running vLLM's offline `LLM().chat()` in-process."""

    name = "offline-vllm"

    def __init__(
        self,
        model: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        dtype: str = "bfloat16",
        max_tokens_default: int = 2048,
        logprobs_default: bool = True,
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_tokens_default = max_tokens_default
        self.logprobs_default = logprobs_default
        self.llm = None
        self._build_messages = None
        self._scale_to_fit = None
        self._prompt_mapping = None

    def start(self):
        from vllm import LLM

        logger.info(f"Loading {self.model} into vLLM offline engine (dtype={self.dtype})...")
        self.llm = LLM(
            model=self.model,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            mm_processor_kwargs=MM_PROCESSOR_KWARGS,
            limit_mm_per_prompt={"image": 1},
        )
        from surya.inference.backends.openai_client import _build_messages
        from surya.inference.prompts import PROMPT_MAPPING
        from surya.inference.util import scale_to_fit

        self._build_messages = _build_messages
        self._scale_to_fit = scale_to_fit
        self._prompt_mapping = PROMPT_MAPPING
        return None

    def stop(self) -> None:
        self.llm = None

    def _sampling_params(self, item):
        from vllm import SamplingParams

        max_tokens = item.max_tokens or self.max_tokens_default
        want_logprobs = item.request_logprobs or self.logprobs_default
        kwargs: dict[str, Any] = dict(temperature=0.0, top_p=0.1, max_tokens=max_tokens)
        if want_logprobs:
            kwargs["logprobs"] = 1
        if item.guided_json is not None:
            kwargs.update(build_structured_outputs(item.guided_json))
        return SamplingParams(**kwargs)

    def generate(self, batch):
        from surya.inference.schema import BatchOutputItem

        if self.llm is None:
            self.start()
        if not batch:
            return []

        conversations = []
        sampling_params = []
        for item in batch:
            prompt = item.prompt or self._prompt_mapping[item.prompt_type]
            image = self._scale_to_fit(item.image)
            conversations.append(self._build_messages(image, prompt))
            sampling_params.append(self._sampling_params(item))

        outputs = self.llm.chat(
            conversations,
            sampling_params,
            chat_template_content_format="openai",
            use_tqdm=False,
        )

        results = []
        for item, out in zip(batch, outputs):
            comp = out.outputs[0]
            results.append(
                BatchOutputItem(
                    raw=comp.text,
                    token_count=len(comp.token_ids),
                    error=False,
                    mean_token_prob=_mean_token_prob(comp),
                    logprobs=None,
                    metadata=item.metadata,
                )
            )
        return results


def make_manager(backend: OfflineVLLMBackend):
    """A SuryaInferenceManager wired to our offline backend (bypassing autodetect)."""
    from surya.inference import SuryaInferenceManager

    manager = SuryaInferenceManager.__new__(SuryaInferenceManager)
    manager.method = backend.name
    manager.backend = backend
    return manager


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def page_text(page: Any) -> str:
    """Reading-order plain text for one OCR page result (ocr branch of serialize_pages)."""
    parts = []
    for b in sorted(page.blocks, key=lambda b: b.reading_order):
        if b.skipped or not b.html:
            continue
        txt = _html_to_text(b.html)
        if txt:
            parts.append(txt)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The experiment: Bag drives the loop
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=None, help="output bucket/prefix (default: <src>/surya)")
    ap.add_argument("--include", default="**/*.jp2")
    ap.add_argument("--limit", type=int, default=None, help="max input pages")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-model-len", type=int, default=18000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16", help="use float16 on T4/Turing")
    args = ap.parse_args()
    out = args.out or f"{args.src}/surya"

    require_cuda()

    def load_engine():
        backend = OfflineVLLMBackend(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
        )
        from surya.recognition import RecognitionPredictor

        predictor = RecognitionPredictor(make_manager(backend))
        state = {"pages": 0, "t0": time.monotonic(), "inf": 0.0}

        def run(items: list[LoadedItem]):
            images = [open_image(it.path) for it in items]
            t = time.monotonic()
            pages = predictor(images, full_page=True)
            state["inf"] += time.monotonic() - t
            state["pages"] += len(items)
            rate = state["pages"] / (time.monotonic() - state["t0"]) * 60
            logger.info(
                f"{state['pages']} pages | {rate:.0f} pages/min overall | "
                f"inference {state['inf']:.0f}s of {time.monotonic() - state['t0']:.0f}s wall"
            )
            outputs = []
            for it, page in zip(items, pages):
                outputs.append((it.key + ".md", page_text(page)))
                outputs.append((it.key + ".json", json.dumps(page.model_dump(mode="json"))))
            return outputs

        return run

    bag = (
        Bag.from_bucket(args.src, include=args.include, limit=args.limit)
        .map_batches(lambda items, run: run(items), batch_size=args.batch_size, setup=load_engine)
        .to_bucket(out)
    )

    t0 = time.monotonic()
    stats = bag.compute(prefetch=args.prefetch)
    wall = time.monotonic() - t0
    print(
        f"\nRESULT: {stats}\n"
        f"wall(incl. listing+resume-scan): {wall:.0f}s | "
        f"{stats.processed / stats.seconds * 60:.0f} pages/min (compute) | "
        f"batch={args.batch_size} prefetch={args.prefetch} model={args.model}",
        flush=True,
    )


if __name__ == "__main__":
    main()
