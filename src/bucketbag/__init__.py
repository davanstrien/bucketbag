"""bucketbag — a minimal, toolz-style helper for Hugging Face bucket files.

Working with bucket-resident data means the same chore in every script: list a prefix,
download a batch of files to local disk, process them, delete them, repeat — without blowing
up disk or leaking temp files. ``huggingface_hub`` has the raw pieces; ``bucketbag`` wraps the
batch-download-process-cleanup loop into one composable verb.

The headline helper is :func:`batched_files` — think ``toolz.partition_all`` for *bucket
files*, except each batch is downloaded to a temp dir and **deleted before the next** (bounded
scratch space), with optional prefetch overlap. It is **file-type agnostic**: a
:class:`LoadedItem` is just a key + a local path + raw bytes, with opt-in lazy ``.image`` /
``.text()`` / ``.json()``.

    from bucketbag import batched_files

    for batch in batched_files("davanstrien/my-bucket", include="images/**/*.jpg", n=20):
        for it in batch:
            do_something(it.path)      # or it.bytes / it.image / it.text() / it.json()
        # <- this batch's files are auto-deleted as the loop advances

This is **not a framework**: no Bag, no executors, no Jobs orchestration. It composes with
plain ``for`` loops and ``toolz``.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
import shutil
import tempfile
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

# Enable xet high-performance mode before huggingface_hub imports (env is read at import time).
# This is the documented, safe transport boost. The big lever for many-small-files throughput —
# raising xet's per-process file-download concurrency — is workload-dependent (great for small
# files, over-subscribes large ones) and import-time-fixed, so it is OPT-IN via boost(), not set
# silently here. Opt out of even this default with BUCKETBAG_NO_XET_TUNE=1.
if os.environ.get("BUCKETBAG_NO_XET_TUNE", "").lower() not in ("1", "true", "yes", "on"):
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import BucketFile, HfApi  # noqa: E402
from toolz import partition_all  # noqa: E402

__all__ = [
    "LoadedItem",
    "iter_keys",
    "batched_files",
    "completed_keys",
    "put_bytes",
    "put_files",
    "put_text",
    "boost",
    "partition_all",
]

__version__ = "0.2.0"

logger = logging.getLogger("bucketbag")

_BUCKET_PREFIX = "hf://buckets/"


def boost(*, file_concurrency: int = 32, high_performance: bool = True) -> None:
    """Raise xet download concurrency for **many-small-files** workloads (e.g. page images).

    Sets ``HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS`` — xet's per-process concurrent-file
    download cap (default 8) — higher. On ~1 MB files this roughly **2.5x'd** throughput in our
    l4x1 benchmark and made the HfApi path beat both raw download and HfFileSystem. Use a **low**
    value (or don't call this) for **large** files: each file's chunk fetches are managed by
    xet's own connection/concurrency machinery, so many large files in flight at once means that
    many files buffering concurrently — over-subscription and a memory/disk blowup.

    Call this **before your first** :func:`batched_files`/download: xet reads these env vars when
    its runtime first initializes (on the first download), so a call beforehand is picked up. It
    does not override env vars you have already exported.

    Note: the file-concurrency var is not documented in ``huggingface_hub`` (best-effort — an
    unknown var is ignored, never an error); ``HF_XET_HIGH_PERFORMANCE`` is documented.
    """
    if high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", str(file_concurrency))


# ---------------------------------------------------------------------------
# bucket id / glob helpers
# ---------------------------------------------------------------------------
def _parse_bucket(bucket: str) -> tuple[str, str]:
    """Split a bucket reference into ``(bucket_id, prefix)``.

    Accepts ``"ns/bucket"``, ``"ns/bucket/some/prefix"`` or
    ``"hf://buckets/ns/bucket/some/prefix"``. ``prefix`` is ``""`` when absent.
    """
    s = bucket
    if s.startswith(_BUCKET_PREFIX):
        s = s[len(_BUCKET_PREFIX) :]
    s = s.strip("/")
    parts = s.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Bucket reference must be 'ns/bucket' (optionally with a prefix), got: {bucket!r}"
        )
    bucket_id = f"{parts[0]}/{parts[1]}"
    prefix = "/".join(parts[2:])
    return bucket_id, prefix


_GLOB_CACHE: dict[str, re.Pattern[str]] = {}


def _glob_to_re(pattern: str) -> re.Pattern[str]:
    """Compile a shell-style glob to a regex with proper path semantics.

    ``*`` matches within a path segment (not ``/``); ``**`` matches across ``/``;
    ``**/`` matches zero or more directories; ``?`` matches one non-``/`` char.
    """
    cached = _GLOB_CACHE.get(pattern)
    if cached is not None:
        return cached
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                j = i + 2
                if pattern[j : j + 1] == "/":
                    out.append("(?:.*/)?")
                    j += 1
                else:
                    out.append(".*")
                i = j
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    compiled = re.compile("(?s:" + "".join(out) + r")\Z")
    _GLOB_CACHE[pattern] = compiled
    return compiled


def _glob_prefix(pattern: str) -> str:
    """Return the literal leading directory of a glob (for a cheap server-side list prefix).

    ``"images/**/*.jp2" -> "images/"``; ``"a/b/c.txt" -> "a/b/"``; ``"*.jp2" -> ""``.
    """
    head = pattern
    for i, ch in enumerate(pattern):
        if ch in "*?[":
            head = pattern[:i]
            break
    slash = head.rfind("/")
    return head[: slash + 1] if slash >= 0 else ""


def _key_of(remote: BucketFile | str) -> str:
    return remote.path if isinstance(remote, BucketFile) else remote


def _resolve_dir(dir: str | os.PathLike[str] | None) -> Path:
    """Pick the download dir. Default: ``/dev/shm`` (RAM tmpfs) if usable, else the system temp."""
    if dir is not None:
        p = Path(dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


# ---------------------------------------------------------------------------
# LoadedItem
# ---------------------------------------------------------------------------
@dataclass
class LoadedItem:
    """One downloaded bucket file, valid only for the lifetime of its batch.

    File-type agnostic: the core is ``key`` + local ``path`` + raw ``.bytes``. The decoders
    ``.image`` / ``.text()`` / ``.json()`` are opt-in (each errs if the bytes aren't that type).

    Do **not** retain a ``LoadedItem`` (or its ``.path``) past its batch — the file is deleted
    when the :func:`batched_files` loop advances.
    """

    key: str
    path: Path
    _bytes: bytes | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def bytes(self) -> bytes:
        """Raw file bytes (read from disk once, then cached)."""
        if self._bytes is None:
            self._bytes = self.path.read_bytes()
        return self._bytes

    @property
    def image(self):
        """Lazily open the file as a ``PIL.Image`` (raises if the bytes aren't an image).

        Each access re-opens the file, and ``Image.open`` is itself lazy — **decode (or
        ``.load()``) within the batch**, before the underlying file is deleted. Formats Pillow
        lacks a codec for (e.g. ``.jp2`` without OpenJPEG) need your own decoder on ``.path``
        (e.g. ``imagecodecs.imread``).
        """
        from PIL import Image

        return Image.open(self.path)

    def text(self, encoding: str = "utf-8") -> str:
        """Decode the bytes as text."""
        return self.bytes.decode(encoding)

    def json(self) -> Any:
        """Parse the bytes as JSON."""
        return _json.loads(self.bytes)


# ---------------------------------------------------------------------------
# iter_keys
# ---------------------------------------------------------------------------
def _list_bucketfiles(
    bucket_id: str,
    *,
    prefix: str | None,
    include: str | None,
    exclude: str | None,
    start_after: str | None,
    limit: int | None,
    token: str | bool | None,
) -> list[BucketFile]:
    """List a bucket prefix, filter by glob / cursor, return sorted ``BucketFile`` objects."""
    list_prefix = prefix
    if list_prefix is None and include:
        list_prefix = _glob_prefix(include) or None
    inc = _glob_to_re(include) if include else None
    exc = _glob_to_re(exclude) if exclude else None

    api = HfApi(token=token)
    api_prefix = list_prefix.rstrip("/") if list_prefix else None
    found: list[BucketFile] = []
    for item in api.list_bucket_tree(bucket_id, prefix=api_prefix or None, recursive=True):
        if getattr(item, "type", None) != "file":
            continue
        key = item.path
        if inc is not None and not inc.match(key):
            continue
        if exc is not None and exc.match(key):
            continue
        if start_after is not None and key <= start_after:
            continue
        found.append(item)  # type: ignore[arg-type]
        # Early-terminate: list_bucket_tree paginates in lexical order, so breaking once we
        # have `limit` matches is both fast (no full-bucket scan) and stable across runs.
        if limit is not None and len(found) >= limit:
            break

    found.sort(key=lambda f: f.path)
    return found


def iter_keys(
    bucket: str,
    *,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    start_after: str | None = None,
    limit: int | None = None,
    token: str | bool | None = None,
) -> Iterator[str]:
    """List keys under a bucket prefix, glob-filtered and sorted (deterministic).

    Args:
        bucket: ``"ns/bucket"`` (optionally with a prefix, or an ``hf://buckets/...`` URL).
        prefix: server-side list prefix; if omitted it is derived from the literal head of
            ``include`` so you only list what you need.
        include / exclude: glob patterns matched against the full key (``**`` crosses ``/``).
        start_after: skip keys ``<= start_after`` (a cheap cursor to start mid-corpus).
        limit: cap the number of keys returned.

    Yields:
        Bucket-relative keys (strings), sorted.
    """
    bucket_id, embedded_prefix = _parse_bucket(bucket)
    effective_prefix = prefix if prefix is not None else (embedded_prefix or None)
    for f in _list_bucketfiles(
        bucket_id,
        prefix=effective_prefix,
        include=include,
        exclude=exclude,
        start_after=start_after,
        limit=limit,
        token=token,
    ):
        yield f.path


# ---------------------------------------------------------------------------
# batched_files — the headline helper
# ---------------------------------------------------------------------------
def _pack(
    remotes: list[BucketFile | str], n: int | None, max_bytes: int | None
) -> Iterator[list[BucketFile | str]]:
    """Group files into batches capped by count (``n``) and/or total bytes (``max_bytes``).

    Whichever cap is hit first ends a batch; a single file larger than ``max_bytes`` forms its own
    batch (files can't be split). Sizes come from ``BucketFile.size``; unknown sizes count as 0.
    """
    batch: list[BucketFile | str] = []
    total = 0
    for r in remotes:
        size = getattr(r, "size", 0) or 0
        full_by_count = n is not None and len(batch) >= n
        full_by_bytes = max_bytes is not None and bool(batch) and total + size > max_bytes
        if full_by_count or full_by_bytes:
            yield batch
            batch, total = [], 0
        batch.append(r)
        total += size
    if batch:
        yield batch


def batched_files(
    bucket: str,
    *,
    keys: Iterable[BucketFile | str] | None = None,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    n: int | None = 20,
    max_bytes: int | None = None,
    dir: str | os.PathLike[str] | None = None,
    prefetch: int = 2,
    max_workers: int | None = None,
    start_after: str | None = None,
    limit: int | None = None,
    token: str | bool | None = None,
) -> Iterator[list[LoadedItem]]:
    """``partition_all`` for bucket files: download each batch, yield it, delete it.

    Lists the bucket (or uses ``keys=``), then downloads files in batches to ``dir`` (default
    ``/dev/shm`` if available, else the system temp) and yields ``list[LoadedItem]``. Each batch's
    files are removed before the loop advances, so the footprint stays bounded. Cleanup is
    guaranteed even on exception — including prefetched lookahead batches when the consumer
    stops early.

    Batch size is capped by **file count** (``n``) and/or **total bytes** (``max_bytes``),
    whichever is hit first. ``max_bytes`` is the better bound for a predictable footprint when
    files vary in size — scratch high-water is then ≈ ``(prefetch + 1) * max_bytes`` regardless
    of file type. NB the default scratch dir is ``/dev/shm`` — **RAM tmpfs, not disk** — so size
    that budget against available memory, or pass ``dir=`` to use real disk. ``max_bytes`` needs
    file sizes, which bucketbag has when it lists for you (or when you pass ``BucketFile``
    objects as ``keys``); with bare string keys ``max_bytes`` is ignored (warned).

    When the helper lists for you it keeps the ``BucketFile`` objects and passes them to
    ``download_bucket_files``, which skips the per-file metadata fetch. (Passing ``keys=`` as
    strings costs one metadata batch per download.)

    Args:
        bucket: ``"ns/bucket"`` (optionally with a prefix, or an ``hf://buckets/...`` URL).
        keys: explicit keys to fetch; if omitted the bucket is listed (``prefix``/``include``/
            ``exclude``/``start_after``/``limit`` apply). Useful for resume: filter out
            :func:`completed_keys` first.
        n: max files per batch (``None`` = no count cap; pair with ``max_bytes`` for pure
            size-based batching).
        max_bytes: max total bytes per batch. A single file larger than this forms its own batch.
        dir: download directory (RAM tmpfs by default).
        prefetch: how many batches to download ahead of the consumer (``0`` = fully sequential).
            Overlaps download I/O with your processing **and** raises concurrent-download
            throughput; also raises disk high-water (≈ ``prefetch + 1`` batches in flight). ``1``
            adds lookahead but no extra download concurrency; use ``>= 2`` to overlap downloads.
        max_workers: thread-pool size for prefetch downloads (defaults to ``prefetch``).
        token: HF token (defaults to the logged-in token).

    Yields:
        ``list[LoadedItem]`` for each batch (already on local disk).
    """
    bucket_id, embedded_prefix = _parse_bucket(bucket)
    if keys is not None:
        remotes: list[BucketFile | str] = list(keys)
    else:
        effective_prefix = prefix if prefix is not None else (embedded_prefix or None)
        remotes = list(
            _list_bucketfiles(
                bucket_id,
                prefix=effective_prefix,
                include=include,
                exclude=exclude,
                start_after=start_after,
                limit=limit,
                token=token,
            )
        )
    if not remotes:
        return

    base_dir = _resolve_dir(dir)
    api = HfApi(token=token)

    def _download_chunk(chunk: list[BucketFile | str]) -> tuple[Path, list[LoadedItem]]:
        tmpdir = Path(tempfile.mkdtemp(dir=base_dir, prefix="bb-"))
        files: list[tuple[BucketFile | str, str]] = []
        items: list[LoadedItem] = []
        for idx, remote in enumerate(chunk):
            key = _key_of(remote)
            local = tmpdir / f"{idx:06d}{PurePosixPath(key).suffix}"
            files.append((remote, str(local)))
            items.append(LoadedItem(key=key, path=local))
        try:
            api.download_bucket_files(bucket_id, files=files, token=token)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        present = [it for it in items if it.path.exists()]
        if len(present) != len(items):
            logger.warning("Skipped %d missing file(s) in batch", len(items) - len(present))
        return tmpdir, present

    if max_bytes is not None and not isinstance(remotes[0], BucketFile):
        logger.warning(
            "max_bytes ignored: file sizes are unknown for string keys "
            "(let batched_files list, or pass BucketFile objects as keys)."
        )
        max_bytes = None
    chunks = list(_pack(remotes, n, max_bytes))

    if prefetch <= 0:
        for chunk in chunks:
            tmpdir, present = _download_chunk(chunk)
            try:
                yield present
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        return

    pool_size = max_workers if max_workers is not None else prefetch
    pool_size = max(1, pool_size)
    in_flight = prefetch + 1
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="bb-dl") as ex:
        futures: deque = deque()
        nxt = 0
        try:
            while nxt < len(chunks) and len(futures) < in_flight:
                futures.append(ex.submit(_download_chunk, chunks[nxt]))
                nxt += 1
            while futures:
                tmpdir, present = futures.popleft().result()
                if nxt < len(chunks):
                    futures.append(ex.submit(_download_chunk, chunks[nxt]))
                    nxt += 1
                try:
                    yield present
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
        finally:
            # Drain the lookahead (issue #2): if the consumer stops early (exception,
            # GeneratorExit) or a download fails mid-loop, batches already fetched — or
            # still in flight — in the prefetch deque must be deleted too, not just the
            # currently-yielded one. A not-yet-started future is cancelled (no tmpdir
            # exists yet); a failed one already removed its own tmpdir in _download_chunk.
            for fut in futures:
                if fut.cancel():
                    continue
                try:
                    leftover_tmpdir, _ = fut.result()
                except Exception:
                    continue
                shutil.rmtree(leftover_tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# completed_keys — cheap resume
# ---------------------------------------------------------------------------
def completed_keys(
    out_bucket: str,
    *,
    prefix: str = "",
    column: str = "__source_key",
    token: str | bool | None = None,
) -> set[str]:
    """Scan parquet outputs and return the set of already-done keys (for cheap resume).

    Reads only ``column`` from every ``.parquet`` object under ``prefix`` in ``out_bucket``.
    Returns an empty set if nothing is there yet.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pyarrow is not a bucketbag dependency
        raise ImportError(
            "completed_keys reads parquet outputs and needs pyarrow, which bucketbag does not "
            "install. Add pyarrow to your script's dependencies (or resume by listing your "
            "outputs with iter_keys and mapping keys back)."
        ) from exc
    from huggingface_hub import HfFileSystem

    bucket_id, embedded_prefix = _parse_bucket(out_bucket)
    list_prefix = prefix or embedded_prefix or None
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)

    done: set[str] = set()
    for item in api.list_bucket_tree(bucket_id, prefix=list_prefix or None, recursive=True):
        if getattr(item, "type", None) != "file" or not item.path.endswith(".parquet"):
            continue
        full = f"{_BUCKET_PREFIX}{bucket_id}/{item.path}"
        try:
            with fs.open(full, "rb") as fh:
                table = pq.read_table(fh, columns=[column])
            done.update(table.column(column).to_pylist())
        except Exception as exc:  # noqa: BLE001 - resume must tolerate a bad/partial shard
            logger.warning("completed_keys: could not read %s (%s)", item.path, exc)
    return done


# ---------------------------------------------------------------------------
# put_bytes / put_text / put_files — per-object writes
# ---------------------------------------------------------------------------
def put_files(
    files: Iterable[tuple[str, bytes | str]],
    out_bucket: str,
    *,
    encoding: str = "utf-8",
    token: str | bool | None = None,
) -> None:
    """Upload many ``(key, content)`` pairs to the bucket in one batch.

    Each ``key`` lands under any prefix embedded in ``out_bucket``; ``str`` content is encoded
    with ``encoding``. One API call for the whole batch — use this (not a :func:`put_bytes`
    loop) when writing a batch of outputs, e.g. one ``.md`` per processed page.
    """
    bucket_id, embedded_prefix = _parse_bucket(out_bucket)
    add: list[tuple[bytes, str]] = []
    for key, content in files:
        data = content.encode(encoding) if isinstance(content, str) else content
        dest = f"{embedded_prefix}/{key}" if embedded_prefix else key
        add.append((data, dest))
    if not add:
        return
    HfApi(token=token).batch_bucket_files(bucket_id, add=add, token=token)


def put_bytes(
    data: bytes,
    out_bucket: str,
    key: str,
    *,
    token: str | bool | None = None,
) -> None:
    """Write one object to the bucket (``key`` under any prefix embedded in ``out_bucket``)."""
    put_files([(key, data)], out_bucket, token=token)


def put_text(
    text: str,
    out_bucket: str,
    key: str,
    *,
    encoding: str = "utf-8",
    token: str | bool | None = None,
) -> None:
    """Write one text object to the bucket (encoded with ``encoding``)."""
    put_files([(key, text)], out_bucket, encoding=encoding, token=token)


# ---------------------------------------------------------------------------
# Bag — lazy plan over the helpers above (imported last: bag.py imports back
# into this module, so the core names must already exist).
# ---------------------------------------------------------------------------
from .bag import Bag, BagStats  # noqa: E402

__all__ += ["Bag", "BagStats"]
