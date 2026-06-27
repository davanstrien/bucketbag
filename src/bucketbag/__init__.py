"""bucketbag — a minimal, toolz-style helper for Hugging Face bucket files.

Working with bucket-resident data means the same chore in every script: list a prefix,
download a batch of files to local disk, process them, delete them, repeat — without blowing
up disk or leaking temp files. ``huggingface_hub`` has the raw pieces; ``bucketbag`` wraps the
batch-download-process-cleanup loop into one composable verb.

The headline helper is :func:`batched_files` — think ``toolz.partition_all`` for *bucket
files*, except each batch is downloaded to a temp dir and **deleted before the next** (bounded
disk), with optional prefetch overlap. It is **file-type agnostic**: a :class:`LoadedItem` is
just a key + a local path + raw bytes, with opt-in lazy ``.image`` / ``.text()`` / ``.json()``.

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

from huggingface_hub import HfApi
from huggingface_hub._buckets import BucketFile
from toolz import partition_all

__all__ = [
    "LoadedItem",
    "iter_keys",
    "batched_files",
    "completed_keys",
    "write_parquet",
    "partition_all",
]

__version__ = "0.1.0"

logger = logging.getLogger("bucketbag")

_BUCKET_PREFIX = "hf://buckets/"


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
        """Lazily open the file as a ``PIL.Image`` (raises if the bytes aren't an image)."""
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
def batched_files(
    bucket: str,
    *,
    keys: Iterable[str] | None = None,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    n: int = 20,
    dir: str | os.PathLike[str] | None = None,
    prefetch: int = 1,
    max_workers: int | None = None,
    start_after: str | None = None,
    limit: int | None = None,
    token: str | bool | None = None,
) -> Iterator[list[LoadedItem]]:
    """``partition_all`` for bucket files: download each batch, yield it, delete it.

    Lists the bucket (or uses ``keys=``), then downloads files in batches of ``n`` to ``dir``
    (default ``/dev/shm`` if available, else the system temp) and yields ``list[LoadedItem]``.
    Each batch's files are removed before the loop advances, so disk stays bounded — to roughly
    ``(prefetch + 1) * n`` files at a time. Cleanup is guaranteed even on exception.

    When the helper lists for you it keeps the ``BucketFile`` objects and passes them to
    ``download_bucket_files``, which skips the per-file metadata fetch. (Passing ``keys=`` as
    strings costs one metadata batch per download.)

    Args:
        bucket: ``"ns/bucket"`` (optionally with a prefix, or an ``hf://buckets/...`` URL).
        keys: explicit keys to fetch; if omitted the bucket is listed (``prefix``/``include``/
            ``exclude``/``start_after``/``limit`` apply). Useful for resume: filter out
            :func:`completed_keys` first.
        n: files per batch.
        dir: download directory (RAM tmpfs by default).
        prefetch: how many batches to download ahead of the consumer (``0`` = fully sequential).
            Overlaps download I/O with your processing; raises disk high-water accordingly.
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

    chunks = [list(c) for c in partition_all(n, remotes)]

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
    import pyarrow.parquet as pq
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
# write_parquet — optional one-shot push
# ---------------------------------------------------------------------------
def write_parquet(
    rows: Iterable[dict[str, Any]],
    out_bucket: str,
    key: str,
    *,
    token: str | bool | None = None,
) -> None:
    """Write ``rows`` (list of dicts) as one parquet object to the bucket.

    The destination is ``key`` under any prefix embedded in ``out_bucket``. Columns are the
    union of the row keys, preserving first-seen order. No-op for empty ``rows``.
    """
    import io as _io

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = list(rows)
    if not rows:
        return

    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                columns.append(k)
    table = pa.Table.from_pydict({c: [row.get(c) for row in rows] for c in columns})

    buf = _io.BytesIO()
    pq.write_table(table, buf, compression="zstd")

    bucket_id, embedded_prefix = _parse_bucket(out_bucket)
    dest = f"{embedded_prefix}/{key}" if embedded_prefix else key
    HfApi(token=token).batch_bucket_files(bucket_id, add=[(buf.getvalue(), dest)], token=token)
