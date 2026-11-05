"""The I/O shell: every hub call bucketbag makes, and the temp files it owns.

Decisions (which keys a listing keeps, how files group into batches, where outputs go) come
from ``_core.py``. This module lists, downloads, uploads and cleans up.
"""

from __future__ import annotations

import builtins
import functools
import json as _json
import logging
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, overload

from huggingface_hub import BucketFile, HfApi

from . import _xet_env
from ._core import (
    BucketRef,
    _check_batch_settings,
    _glob_prefix,
    _glob_to_re,
    _group_batches,
    _keeps,
    _pack,
)

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger("bucketbag")

_BUCKET_SCHEME = "hf://buckets/"


def boost(*, file_concurrency: int = 32, high_performance: bool = True) -> None:
    """Set xet's concurrent file downloads for this process (and high-performance mode).

    bucketbag already sets 32 concurrent files and high-performance mode when it is imported
    (unless you exported your own values or set ``BUCKETBAG_NO_XET_TUNE=1``). Call this to
    pick another file count. It replaces bucketbag's import-time defaults, never a value you
    exported yourself.

    Call it **before the first download** in the process: xet reads these variables when its
    runtime starts. ``HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS`` is xet's per-process limit
    (xet's default is 8), shared by every download call.
    """
    if high_performance:
        _xet_env.set_value(_xet_env.HIGH_PERFORMANCE, "1")
    _xet_env.set_value(_xet_env.FILE_CONCURRENCY, str(file_concurrency))


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
@dataclass(frozen=True)
class LoadedItem:
    """One downloaded bucket file, valid only for the lifetime of its batch.

    File-type agnostic: the core is ``key`` + local ``path`` + raw ``.bytes``. The decoders
    ``.image`` / ``.text()`` / ``.json()`` are opt-in (each errs if the bytes aren't that type).

    Do **not** retain a ``LoadedItem`` (or its ``.path``) past its batch — the file is deleted
    when the :func:`batched_files` loop advances.
    """

    key: str
    path: Path

    # `builtins.bytes`: inside the class body, `bytes` names this property.
    @functools.cached_property
    def bytes(self) -> builtins.bytes:
        """Raw file bytes (read from disk once, then cached)."""
        return self.path.read_bytes()

    @property
    def image(self) -> Image:
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
    ref: BucketRef,
    *,
    include: str | None,
    exclude: str | None,
    start_after: str | None,
    limit: int | None,
    token: str | bool | None,
) -> list[BucketFile]:
    """List the files under ``ref`` that the filters keep, sorted by key.

    The prefix is sent to the Hub verbatim, with a significant trailing slash; ``_keeps`` checks
    it again on every key. Without a prefix, the literal head of ``include`` narrows the listing.
    """
    prefix = ref.prefix or (_glob_prefix(include) if include else "")
    inc = _glob_to_re(include) if include else None
    exc = _glob_to_re(exclude) if exclude else None
    found: list[BucketFile] = []
    listing = HfApi(token=token).list_bucket_tree(
        ref.bucket_id, prefix=prefix or None, recursive=True
    )
    for item in listing:
        if not isinstance(item, BucketFile) or not _keeps(item.path, prefix, inc, exc, start_after):
            continue
        found.append(item)
        # `limit` stops the listing early. The Hub lists in lexical key order today but does
        # not document it (issue #9); the sort below cannot add keys it had not returned yet.
        if limit is not None and len(found) >= limit:
            break
    found.sort(key=lambda f: f.path)
    return found


@overload
def iter_keys(
    bucket: str,
    *,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    start_after: str | None = None,
    limit: int | None = None,
    objects: Literal[False] = False,
    token: str | bool | None = None,
) -> Iterator[str]: ...


@overload
def iter_keys(
    bucket: str,
    *,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    start_after: str | None = None,
    limit: int | None = None,
    objects: Literal[True],
    token: str | bool | None = None,
) -> Iterator[BucketFile]: ...


def iter_keys(
    bucket: str,
    *,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    start_after: str | None = None,
    limit: int | None = None,
    objects: bool = False,
    token: str | bool | None = None,
) -> Iterator[BucketFile] | Iterator[str]:
    """List keys under a bucket prefix, glob-filtered and sorted (deterministic).

    Args:
        bucket: ``"ns/bucket"``, ``"ns/bucket/prefix"``, ``"hf://buckets/ns/bucket/prefix"`` or
            a Hub web URL. A trailing ``/`` on the prefix limits it to one directory.
        prefix: string prefix keys must start with, joined below any prefix in ``bucket`` as a
            directory (``"ns/b/out"`` with ``prefix="part-"`` lists ``out/part-*``). A trailing
            ``/`` is significant: ``"a/b/"`` lists the directory ``a/b`` only, while ``"a/b"``
            also matches siblings such as ``a/bc/y``. Without any prefix, the literal head of
            ``include`` narrows the listing.
        include / exclude: globs matched against the full key from the bucket root, with
            fsspec's rules (``*`` within a segment, ``**`` across ``/``).
        start_after: skip keys ``<= start_after`` (a cheap cursor to start mid-corpus).
        limit: cap the number of keys returned. Listing stops after the first ``limit``
            matches in the server's pagination order. The Hub paginates in lexical key order
            today, so this gives the lexically first ``limit`` keys, but the Hub does not
            document that order.
        objects: yield :class:`~huggingface_hub.BucketFile` objects (carrying ``.size``)
            instead of key strings. Needed to feed :func:`batched_files` ``keys=`` together
            with ``max_bytes`` — string keys carry no size, so ``max_bytes`` cannot be honored.

    Yields:
        Full keys from the bucket root (strings by default; ``BucketFile`` objects with
        ``objects=True``), sorted.
    """
    ref = BucketRef.parse(bucket).under(prefix)
    for f in _list_bucketfiles(
        ref, include=include, exclude=exclude, start_after=start_after, limit=limit, token=token
    ):
        yield f if objects else f.path


# ---------------------------------------------------------------------------
# batched_files — the headline helper
# ---------------------------------------------------------------------------
# The smallest download call: consecutive batches share a call until it holds this many bytes
# or files. Every download_bucket_files call costs about 0.4 s before data moves (a metadata
# request and a new xet download group, which fetches a CAS token), so small batches of small
# files spent most of their time on it. Batches at or above either size are a call each, as
# before. The file count keeps the first batch of tiny files from waiting on thousands.
_CALL_BYTES = 64_000_000
_CALL_FILES = 256
# Download threads when prefetch is not set. Calls beyond this wait in the pool's queue.
_THREADS = 8
# The byte budget when prefetch_bytes is not set: 2 GB, or room for this many of the largest
# call when that is more (the call in use and two ahead, as 0.4.0's prefetch=2 kept). A 2 GB
# budget alone keeps one call ahead of a 2 GB batch, which was slower on 0.5 GB files.
_DEFAULT_BUDGET = 2e9
_DEFAULT_CALLS = 3


@dataclass(frozen=True)
class _Call:
    """One finished download call: its temp dir, and one entry per requested file.

    ``items[i]`` is ``None`` when the i-th requested file was missing at download time.
    """

    tmpdir: Path
    items: list[LoadedItem | None]


def _download_call(
    api: HfApi,
    bucket_id: str,
    chunk: list[BucketFile | str],
    base_dir: Path,
    token: str | bool | None,
) -> _Call:
    """Download ``chunk`` with one hub call into a new temp dir under ``base_dir``.

    The caller owns the returned ``_Call.tmpdir``. If the download fails, the dir is already gone.
    """
    tmpdir = Path(tempfile.mkdtemp(dir=base_dir, prefix="bb-"))
    files: list[tuple[str | BucketFile, str | Path]] = []
    items: list[LoadedItem] = []
    for idx, remote in enumerate(chunk):
        key = remote.path if isinstance(remote, BucketFile) else remote
        local = tmpdir / f"{idx:06d}{PurePosixPath(key).suffix}"
        files.append((remote, str(local)))
        items.append(LoadedItem(key=key, path=local))
    try:
        api.download_bucket_files(bucket_id, files=files, token=token)
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    present: list[LoadedItem | None] = [it if it.path.exists() else None for it in items]
    missing = present.count(None)
    if missing:
        logger.warning("Skipped %d missing file(s) in batch", missing)
    return _Call(tmpdir, present)


def _download_calls(
    remotes: Sequence[BucketFile | str],
    n: int | None,
    max_bytes: float | None,
    budget: float | None,
) -> Iterator[list[BucketFile | str]]:
    """Group batches into download calls: small ones together when sizes are known.

    A call holds about ``min(64 MB, budget / 4)``, so a small budget still keeps several calls
    in flight. String keys carry no size, so each of their batches is one call.
    """
    batches = _pack(remotes, n, max_bytes)
    if budget is None:
        return batches
    return _group_batches(batches, min(_CALL_BYTES, budget / 4), _CALL_FILES)


def _call_bytes(call: list[BucketFile | str]) -> int:
    return sum(getattr(r, "size", 0) or 0 for r in call)


def _warn_if_a_call_cannot_fit(calls: Iterable[list[BucketFile | str]], base_dir: Path) -> None:
    """Warn when the largest download call is larger than the scratch dir's free space.

    The byte budget is already capped at half the free space, but one call always runs, so a
    batch larger than the free space cannot fit. Only sized keys count.
    """
    largest = max((_call_bytes(call) for call in calls), default=0)
    free = shutil.disk_usage(base_dir).free
    if largest > free:
        logger.warning(
            "One download call needs %.2f GB on %s, but only %.2f GB is free. Lower max_bytes "
            "(or n / batch_size), or pass dir= on a larger disk. /dev/shm is RAM, so running "
            "out there kills the process.",
            largest / 1e9,
            base_dir,
            free / 1e9,
        )


def _drain(ahead: deque[Future[_Call]]) -> None:
    """Delete every queued call: cancel downloads not started, then wait for the rest.

    Cancelling first means no queued download starts while we wait. ``exception()`` waits
    without raising a worker's error (a failed download already removed its own dir), so a
    Ctrl-C during the wait still reaches the caller.
    """
    running = [future for future in ahead if not future.cancel()]
    for future in running:
        if future.exception() is None:
            shutil.rmtree(future.result().tmpdir, ignore_errors=True)


def batched_files(
    bucket: str,
    *,
    keys: Iterable[BucketFile | str] | None = None,
    prefix: str | None = None,
    include: str | None = None,
    exclude: str | None = None,
    n: int | None = 20,
    max_bytes: float | None = None,
    dir: str | os.PathLike[str] | None = None,
    prefetch: int | None = None,
    prefetch_bytes: float | None = None,
    start_after: str | None = None,
    limit: int | None = None,
    token: str | bool | None = None,
) -> Generator[list[LoadedItem], None, None]:
    """``partition_all`` for bucket files: download each batch, yield it, delete it.

    Lists the bucket (or uses ``keys=``), then downloads files to ``dir`` (default ``/dev/shm``
    if available, else the system temp) and yields ``list[LoadedItem]`` batches in key order.

    Every hub call has a fixed cost, so small batches share one: consecutive batches go into one
    download call until it holds about 64 MB (less with a small ``prefetch_bytes``) or 256 files.
    A batch at or above that size is a call of its own. String keys carry no size, so with
    string keys each batch is one call.

    Each call's temp dir has one owner: the download worker until it returns, then this
    generator, which deletes a batch's files when the loop moves past it and the dir once all
    its files are used. Calls downloaded ahead are deleted too when the loop body raises or the
    generator is closed. A ``for`` loop closes it when the loop ends; a caller that keeps the
    generator in a variable must call ``.close()``.

    Downloads run ahead of the batch you are processing while the files on disk and on their way
    stay within a byte budget, at most half the scratch dir's free space. By default the budget
    is 2 GB, or three of the largest download call when that is more (the call in use and two
    ahead), so big batches keep their overlap. ``prefetch_bytes`` sets a hard budget instead: use
    it on a small machine. One call always downloads ahead of the batch in use, so a batch larger
    than a hard budget still overlaps with the next (scratch then holds about two calls); ``0``
    downloads one call at a time, with no overlap. NB the default scratch dir is ``/dev/shm`` —
    **RAM tmpfs, not disk** — so the budget comes out of memory unless you pass ``dir=`` on real
    disk.

    ``max_bytes`` needs file sizes, which bucketbag has when it lists for you (or when you pass
    ``BucketFile`` objects as ``keys``, e.g. from ``iter_keys(..., objects=True)``); bare string
    keys with ``max_bytes`` raise ``ValueError``.

    When the helper lists for you it keeps the ``BucketFile`` objects and passes them to
    ``download_bucket_files``, which skips the per-file metadata fetch. (Passing ``keys=`` as
    strings costs one metadata batch per download.)

    Args:
        bucket: as in :func:`iter_keys`.
        keys: explicit keys to fetch; if omitted the bucket is listed (``prefix``/``include``/
            ``exclude``/``start_after``/``limit`` apply, as in :func:`iter_keys`).
        prefix: as in :func:`iter_keys`.
        n: max files per batch (``None`` = no count cap; pair with ``max_bytes`` for pure
            size-based batching). At least one of ``n`` and ``max_bytes`` must be set.
        max_bytes: max total bytes per batch (any finite number). A single file larger than
            this forms its own batch.
        dir: download directory (RAM tmpfs by default).
        prefetch: optional cap on the download calls ahead of the batch you are processing
            (``0`` = one call at a time); it also sets the number of download threads (default
            8). With string keys, which carry no size, it is the only bound (default 2).
        prefetch_bytes: a hard limit on the bytes of downloads on disk and on their way, capped
            at half the scratch dir's free space. Default ``None``: 2 GB, or three of the
            largest download call when that is more. Ignored for string keys.
        token: HF token (defaults to the logged-in token).

    Yields:
        ``list[LoadedItem]`` for each batch (already on local disk).
    """
    _check_batch_settings(n, max_bytes, prefetch, prefetch_bytes)
    ref = BucketRef.parse(bucket)
    if keys is not None:
        remotes: Sequence[BucketFile | str] = list(keys)
    else:
        remotes = _list_bucketfiles(
            ref.under(prefix),
            include=include,
            exclude=exclude,
            start_after=start_after,
            limit=limit,
            token=token,
        )
    if max_bytes is not None and not all(isinstance(r, BucketFile) for r in remotes):
        raise ValueError(
            "max_bytes requires sized keys, but string keys were given (sizes are unknown, "
            "so the byte bound could not be honored). Pass BucketFile objects as keys "
            "(e.g. iter_keys(..., objects=True)), let batched_files do the listing (omit keys=), "
            "or drop max_bytes."
        )
    if not remotes:
        return

    _xet_env.log_once(logger)
    api = HfApi(token=token)
    base_dir = _resolve_dir(dir)
    window = _Bounds.for_run(remotes, n, max_bytes, prefetch, prefetch_bytes, base_dir)
    if window.budget is not None:  # only sized keys can warn
        _warn_if_a_call_cannot_fit(_download_calls(remotes, n, max_bytes, window.budget), base_dir)
    yield from _stream_batches(api, ref.bucket_id, remotes, n, max_bytes, window, base_dir, token)


@dataclass(frozen=True)
class _Bounds:
    """How far downloads may run ahead: a byte budget, a call count, and the thread count."""

    budget: float | None  # None for string keys: their sizes are unknown
    max_calls: int | None  # calls ahead of the one in use; None = the budget decides
    threads: int

    @classmethod
    def for_run(  # noqa: PLR0913 - the run's settings, passed through from batched_files
        cls,
        remotes: Sequence[BucketFile | str],
        n: int | None,
        max_bytes: float | None,
        prefetch: int | None,
        prefetch_bytes: float | None,
        base_dir: Path,
    ) -> _Bounds:
        threads = max(1, prefetch) if prefetch is not None else _THREADS
        if not all(isinstance(r, BucketFile) for r in remotes):
            return cls(None, prefetch if prefetch is not None else 2, threads)
        half_free = shutil.disk_usage(base_dir).free / 2
        if prefetch_bytes is not None:
            return cls(min(prefetch_bytes, half_free), prefetch, threads)
        calls = _download_calls(remotes, n, max_bytes, _DEFAULT_BUDGET)
        largest = max((_call_bytes(call) for call in calls), default=0)
        budget = min(max(_DEFAULT_BUDGET, _DEFAULT_CALLS * largest), half_free)
        if budget > _DEFAULT_BUDGET:
            logger.info(
                "Download budget: %.2f GB, room for the batch in use and two downloading "
                "ahead (the largest download call is %.2f GB; at most half the free space on "
                "%s). Pass prefetch_bytes= for a hard limit.",
                budget / 1e9,
                largest / 1e9,
                base_dir,
            )
        return cls(budget, prefetch, threads)


class _CallWindow:
    """Download calls running ahead, and finished calls whose files are not all used yet.

    Calls are queued while the bytes on disk and on their way stay within the budget (and,
    when ``max_calls`` is set, while at most ``max_calls + 1`` calls are on disk). ``take`` hands
    out the next files in key order, waiting for calls as needed; ``release`` deletes them once
    their batch is done, and a call's dir once all its files are.
    """

    def __init__(
        self,
        pool: ThreadPoolExecutor,
        fetch: Callable[[list[BucketFile | str]], _Call],
        calls: Iterator[list[BucketFile | str]],
        bounds: _Bounds,
    ) -> None:
        self.pool = pool
        self.fetch = fetch
        self.calls = calls
        self.bounds = bounds
        self.next_call: list[BucketFile | str] | None = next(calls, None)
        self.ahead: deque[tuple[Future[_Call], int]] = deque()  # submitted calls + their bytes
        self.held: deque[tuple[_Call, int, int]] = deque()  # finished calls + end index + bytes
        self.ready: deque[LoadedItem | None] = deque()  # their files not yet taken, in key order
        self.bytes = 0  # bytes of the calls in ahead and held
        self.received = 0  # files received from finished calls
        self.used = 0  # files released

    def _has_room(self, size: int) -> bool:
        calls = len(self.ahead) + len(self.held)
        if self.bounds.max_calls is not None and calls >= self.bounds.max_calls + 1:
            return False
        if self.bounds.budget is None:
            return True
        if self.bounds.budget > 0 and not self.ahead:
            return True  # keep one call downloading ahead, so big batches still overlap
        return self.bytes + size <= self.bounds.budget

    def submit(self, *, force: bool = False) -> None:
        """Queue calls while the bounds allow; ``force`` queues one anyway."""
        while self.next_call is not None and (force or self._has_room(_call_bytes(self.next_call))):
            size = _call_bytes(self.next_call)
            self.ahead.append((self.pool.submit(self.fetch, self.next_call), size))
            self.bytes += size
            self.next_call = next(self.calls, None)
            force = False

    def take(self, count: int) -> list[LoadedItem | None]:
        """The next ``count`` files in key order (``None`` for a missing one)."""
        while len(self.ready) < count:
            if not self.ahead:
                self.submit(force=True)  # the batch needs a call the disk bound held back
            # Wait before popping: a Ctrl-C during the wait leaves the future queued, so close()
            # still deletes its files.
            call = self.ahead[0][0].result()
            _future, size = self.ahead.popleft()  # from here the window owns call.tmpdir
            self.received += len(call.items)
            self.held.append((call, self.received, size))
            self.ready.extend(call.items)
            self.submit()  # start the next call while this one's files are in use
        return [self.ready.popleft() for _ in range(count)]

    def release(self, taken: list[LoadedItem | None]) -> None:
        """Delete the files of a finished batch, and every call dir it emptied."""
        for item in taken:
            if item is not None:
                item.path.unlink(missing_ok=True)
        self.used += len(taken)
        while self.held and self.held[0][1] <= self.used:
            call, _end, size = self.held.popleft()
            shutil.rmtree(call.tmpdir, ignore_errors=True)
            self.bytes -= size

    def close(self) -> None:
        """Delete everything still on disk or on its way."""
        _drain(deque(future for future, _size in self.ahead))
        for call, _end, _size in self.held:
            shutil.rmtree(call.tmpdir, ignore_errors=True)
        self.held.clear()


def _stream_batches(  # noqa: PLR0913 - the loop's settings, passed through from batched_files
    api: HfApi,
    bucket_id: str,
    remotes: Sequence[BucketFile | str],
    n: int | None,
    max_bytes: float | None,
    bounds: _Bounds,
    base_dir: Path,
    token: str | bool | None,
) -> Generator[list[LoadedItem], None, None]:
    """Run download calls ahead on a thread pool; yield batches cut from their files in order."""

    def fetch(chunk: list[BucketFile | str]) -> _Call:
        return _download_call(api, bucket_id, chunk, base_dir, token)

    calls = _download_calls(remotes, n, max_bytes, bounds.budget)
    with ThreadPoolExecutor(max_workers=bounds.threads, thread_name_prefix="bb-dl") as pool:
        window = _CallWindow(pool, fetch, calls, bounds)
        try:
            window.submit()
            for batch in _pack(remotes, n, max_bytes):
                taken = window.take(len(batch))
                try:
                    yield [item for item in taken if item is not None]
                finally:
                    window.release(taken)
                window.submit()
        finally:
            window.close()


# ---------------------------------------------------------------------------
# completed_keys — cheap resume from parquet outputs
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
    Returns an empty set if nothing is there yet. A shard that cannot be read is skipped with a
    warning; a readable shard without ``column`` raises ``ValueError``.

    ``prefix`` joins below any prefix in ``out_bucket``, as in :func:`iter_keys`, and its
    trailing ``/`` is significant: ``"out/run1/"`` scans that directory only, ``"out/run1"``
    also scans a sibling ``out/run10/``.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pyarrow is not a bucketbag dependency
        raise ImportError(
            "completed_keys reads parquet outputs and needs pyarrow, which bucketbag does not "
            "install. Add pyarrow to your script's dependencies (or resume by listing your "
            "outputs with iter_keys and mapping keys back)."
        ) from exc
    # Imported here, not at the top: the tests replace huggingface_hub.HfFileSystem.
    from huggingface_hub import HfFileSystem

    ref = BucketRef.parse(out_bucket).under(prefix)
    fs = HfFileSystem(token=token)
    shards = _list_bucketfiles(
        ref, include="**/*.parquet", exclude=None, start_after=None, limit=None, token=token
    )
    done: set[str] = set()
    for shard_file in shards:
        url = f"{_BUCKET_SCHEME}{ref.bucket_id}/{shard_file.path}"
        # pyarrow raises the same error type for a missing column and a corrupt shard, so the
        # schema is checked first: only an unreadable shard is skipped.
        try:
            with fs.open(url, "rb") as fh:
                shard = pq.ParquetFile(fh)
                names = shard.schema_arrow.names
                table = shard.read(columns=[column]) if column in names else None
        except Exception as exc:  # noqa: BLE001 - resume must tolerate a bad/partial shard
            logger.warning("completed_keys: could not read %s (%s)", shard_file.path, exc)
            continue
        if table is None:
            raise ValueError(
                f"completed_keys: {shard_file.path} has no column {column!r} (it has {names}). "
                "Pass the column that holds your source keys as column=."
            )
        done.update(table.column(column).to_pylist())
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

    Each ``key`` lands under any prefix in ``out_bucket``, joined as a directory; ``str``
    content is encoded with ``encoding``. One API call for the whole batch — use this (not a
    :func:`put_bytes` loop) when writing a batch of outputs, e.g. one ``.md`` per processed page.
    """
    ref = BucketRef.parse(out_bucket)
    add: list[tuple[str | Path | bytes, str]] = []
    for key, content in files:
        data = content.encode(encoding) if isinstance(content, str) else content
        add.append((data, ref.key_for(key)))
    if not add:
        return
    HfApi(token=token).batch_bucket_files(ref.bucket_id, add=add, token=token)


def put_bytes(
    data: bytes,
    out_bucket: str,
    key: str,
    *,
    token: str | bool | None = None,
) -> None:
    """Write one object to the bucket (``key`` under any prefix in ``out_bucket``)."""
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
