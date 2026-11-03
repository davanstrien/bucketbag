"""``Bag`` — a lazy, dask.bag-style plan over bucket files (P0: local execution only).

The chain is a *plan*, not work: nothing lists or downloads until :meth:`Bag.compute` /
:meth:`Bag.take`. One shape, three verbs::

    from bucketbag import Bag

    bag = (Bag.from_bucket(SRC, include="**/*.jp2")
             .map_batches(ocr_batch, batch_size=32, setup=load_model)
             .to_bucket(OUT))
    bag.take(3)        # dev: run the fn on the first few items, return outputs, write nothing
    bag.compute()      # run: list -> skip done -> batched download -> fn -> put_files, bounded

The user fn receives a ``list[LoadedItem]`` (plus the ``setup()`` result if given) and returns
an iterable of ``(key, content)`` pairs — content is ``bytes`` or ``str``. Outputs land under
the ``to_bucket`` prefix in one API call per batch.

**Resume contract:** an input is "done" iff some output key *starts with* its source key
(e.g. ``pages/p1.jp2`` -> ``pages/p1.jp2.md``). Emit output keys prefixed by ``item.key`` and
``compute(resume=True)`` (the default) skips finished inputs on re-run — crash-safe long runs
with no bookkeeping.

This is the P0 slice of the bucketbag design: no Jobs fan-out, no shards, no executors —
``compute()`` runs in-process. The fan-out layer (``JobsExecutor``) builds on exactly this
plan object later.
"""

from __future__ import annotations

import bisect
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, TypeVar, overload

from huggingface_hub import BucketFile

from . import (
    LoadedItem,
    _check_batch_settings,
    _list_bucketfiles,
    _parse_bucket,
    batched_files,
    iter_keys,
    logger,
    put_files,
)

__all__ = ["Bag", "BagStats"]

Output = tuple[str, bytes | str]
# fn(items) -> Iterable[Output], or fn(items, ctx) when setup= is given.
MapFn = Callable[..., Iterable[Output]]
C = TypeVar("C")


@dataclass(frozen=True)
class BagStats:
    """What a :meth:`Bag.compute` run did."""

    processed: int
    skipped: int
    batches: int
    outputs: int
    seconds: float

    def __str__(self) -> str:  # human one-liner for print(bag.compute())
        return (
            f"{self.processed} processed (+{self.skipped} skipped as done) in "
            f"{self.batches} batches -> {self.outputs} outputs, {self.seconds:.1f}s"
        )


def _with_context(
    fn: MapFn, setup: Callable[[], Any] | None
) -> Callable[[list[LoadedItem]], Iterable[Output]]:
    """Return the map function as ``f(items)``: ``fn`` itself, or ``fn(items, setup())``.

    ``setup`` runs here, once. The function's signature is never inspected.
    """
    if setup is None:
        return fn
    ctx = setup()
    return lambda items: fn(items, ctx)


def _unmatched_output(pairs: list[Output], items: list[LoadedItem]) -> str | None:
    """The first output key that starts with no input key of its batch, else ``None``.

    Looks up each prefix of the output key in a set, so a large batch stays cheap.
    """
    inputs = {item.key for item in items}
    for key, _ in pairs:
        if not any(key[:end] in inputs for end in range(len(key), 0, -1)):
            return key
    return None


def _warn_if_unmatched(pairs: list[Output], items: list[LoadedItem]) -> bool:
    """Log a warning if an output key cannot mark its input done; return whether it did."""
    orphan = _unmatched_output(pairs, items)
    if orphan is None:
        return False
    logger.warning(
        "Output key %r does not start with an input key of its batch, so resume cannot mark "
        "that input done and it runs again on every re-run. Name each output "
        "<input key><suffix>, e.g. 'pages/p1.jp2.md'.",
        orphan,
    )
    return True


@dataclass(frozen=True)
class Bag:
    """An immutable, lazy plan: source spec + one map stage + output prefix."""

    src: str
    include: str | None = None
    exclude: str | None = None
    limit: int | None = None
    token: str | bool | None = None
    fn: MapFn | None = None
    setup: Callable[[], Any] | None = None
    batch_size: int | None = 32
    max_bytes: float | None = None
    out: str | None = None

    # -- constructors / transforms (each returns a NEW Bag) --------------------
    @classmethod
    def from_bucket(
        cls,
        bucket: str,
        *,
        include: str | None = None,
        exclude: str | None = None,
        limit: int | None = None,
        token: str | bool | None = None,
    ) -> Bag:
        """Start a plan over ``bucket`` (``ns/bucket[/prefix]`` or ``hf://buckets/...``)."""
        return cls(src=bucket, include=include, exclude=exclude, limit=limit, token=token)

    @overload
    def map_batches(
        self,
        fn: Callable[[list[LoadedItem]], Iterable[Output]],
        *,
        batch_size: int | None = 32,
        max_bytes: float | None = None,
        setup: None = None,
    ) -> Bag: ...

    @overload
    def map_batches(
        self,
        fn: Callable[[list[LoadedItem], C], Iterable[Output]],
        *,
        batch_size: int | None = 32,
        max_bytes: float | None = None,
        setup: Callable[[], C],
    ) -> Bag: ...

    def map_batches(
        self,
        fn: MapFn,
        *,
        batch_size: int | None = 32,
        max_bytes: float | None = None,
        setup: Callable[[], Any] | None = None,
    ) -> Bag:
        """Set the per-batch function: ``fn(items) -> iterable[(key, bytes|str)]``.

        With ``setup`` (e.g. a model loader), ``setup()`` runs once per ``compute()`` or
        ``take()``, and ``fn`` is called as ``fn(items, ctx)`` with its result.
        ``batch_size``/``max_bytes`` bound each batch by file count/total bytes; at least one
        must be set, and each must be at least 1.
        """
        _check_batch_settings(batch_size, max_bytes)
        return replace(self, fn=fn, setup=setup, batch_size=batch_size, max_bytes=max_bytes)

    def to_bucket(self, out: str) -> Bag:
        """Set the output bucket/prefix for the fn's ``(key, content)`` pairs."""
        return replace(self, out=out)

    # -- execution --------------------------------------------------------------
    def _list_inputs(self, *, limit: int | None) -> list[BucketFile]:
        """List source inputs as ``BucketFile`` objects (sizes -> max_bytes + fast download)."""
        bucket_id, embedded = _parse_bucket(self.src)
        return _list_bucketfiles(
            bucket_id,
            prefix=embedded or None,
            include=self.include,
            exclude=self.exclude,
            start_after=None,
            limit=limit,
            token=self.token,
        )

    def _list_todo(self, *, resume: bool) -> tuple[list[BucketFile], int]:
        files = self._list_inputs(limit=self.limit)
        if not (resume and self.out):
            return files, 0
        # Output keys come back as full bucket paths; put_files prepended any prefix embedded
        # in `out`, so strip it symmetrically before matching against source keys.
        _, out_prefix = _parse_bucket(self.out)
        strip = f"{out_prefix}/" if out_prefix else ""
        out_keys = sorted(
            k[len(strip) :] for k in iter_keys(self.out, token=self.token) if k.startswith(strip)
        )

        def is_done(key: str) -> bool:
            i = bisect.bisect_left(out_keys, key)
            return i < len(out_keys) and out_keys[i].startswith(key)

        todo = [f for f in files if not is_done(f.path)]
        return todo, len(files) - len(todo)

    def take(self, n: int, *, dir: str | os.PathLike[str] | None = None) -> list[Output]:
        """Dev peek: run the plan on the first ``n`` items and return the outputs. No writes."""
        fn = self.fn
        if fn is None:
            raise ValueError("take() needs a map_batches(fn) stage")
        files = self._list_inputs(limit=n)
        apply = _with_context(fn, self.setup)
        outputs: list[Output] = []
        warned = False
        for batch in batched_files(
            self.src,
            keys=files,
            n=self.batch_size,
            max_bytes=self.max_bytes,
            prefetch=0,
            dir=dir,
            token=self.token,
        ):
            pairs = list(apply(batch))
            warned = warned or _warn_if_unmatched(pairs, batch)
            outputs.extend(pairs)
        return outputs

    def compute(
        self,
        *,
        resume: bool = True,
        prefetch: int = 2,
        dir: str | os.PathLike[str] | None = None,
    ) -> BagStats:
        """Run the plan in-process: list, skip done (``resume``), download, map, write.

        Bounded scratch (via :func:`batched_files`), one ``put_files`` call per batch, and
        resume-by-output-existence — safe to kill and re-run.
        """
        fn = self.fn
        if fn is None:
            raise ValueError("compute() needs a map_batches(fn) stage")
        if self.out is None:
            raise ValueError("compute() needs a to_bucket(out) stage")
        _check_batch_settings(self.batch_size, self.max_bytes, prefetch)

        todo, skipped = self._list_todo(resume=resume)
        if skipped:
            logger.info("Bag.compute: %d inputs already done, %d to do", skipped, len(todo))
        t0 = time.monotonic()
        apply = _with_context(fn, self.setup)
        processed = batches = outputs = 0
        warned = False
        for batch in batched_files(
            self.src,
            keys=todo,
            n=self.batch_size,
            max_bytes=self.max_bytes,
            prefetch=prefetch,
            dir=dir,
            token=self.token,
        ):
            pairs = list(apply(batch))
            warned = warned or _warn_if_unmatched(pairs, batch)
            put_files(pairs, self.out, token=self.token)
            processed += len(batch)
            batches += 1
            outputs += len(pairs)
        return BagStats(
            processed=processed,
            skipped=skipped,
            batches=batches,
            outputs=outputs,
            seconds=time.monotonic() - t0,
        )
