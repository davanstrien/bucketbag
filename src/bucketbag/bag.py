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
import inspect
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from huggingface_hub import BucketFile

from . import _list_bucketfiles, _parse_bucket, batched_files, iter_keys, logger, put_files

__all__ = ["Bag", "BagStats"]

# fn(items) -> Iterable[(key, bytes|str)], or fn(items, ctx) when setup= is given.
MapFn = Callable[..., Iterable[tuple[str, "bytes | str"]]]


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


def _call_fn(fn: MapFn, items: list, ctx: Any, has_setup: bool) -> Iterable[tuple[str, Any]]:
    if has_setup:
        return fn(items, ctx)
    # Allow fn(items) or fn(items, ctx=None) signatures without setup.
    try:
        sig_params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C callables
        return fn(items)
    return fn(items, None) if len(sig_params) >= 2 else fn(items)


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
    max_bytes: int | None = None
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

    def map_batches(
        self,
        fn: MapFn,
        *,
        batch_size: int | None = 32,
        max_bytes: int | None = None,
        setup: Callable[[], Any] | None = None,
    ) -> Bag:
        """Set the per-batch fn: ``fn(items[, ctx]) -> iterable[(key, bytes|str)]``.

        ``setup`` (e.g. a model loader) runs **once per compute** and its result is passed to
        every ``fn`` call. ``batch_size``/``max_bytes`` bound each batch by count/total bytes.
        """
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

    def take(self, n: int, *, dir: str | None = None) -> list[tuple[str, Any]]:
        """Dev peek: run the plan on the first ``n`` items and return the outputs. No writes."""
        if self.fn is None:
            raise ValueError("take() needs a map_batches(fn) stage")
        files = self._list_inputs(limit=n)
        ctx = self.setup() if self.setup is not None else None
        outputs: list[tuple[str, Any]] = []
        for batch in batched_files(
            self.src, keys=files, n=self.batch_size, prefetch=0, dir=dir, token=self.token
        ):
            outputs.extend(_call_fn(self.fn, batch, ctx, self.setup is not None))
        return outputs

    def compute(
        self,
        *,
        resume: bool = True,
        prefetch: int = 2,
        dir: str | None = None,
    ) -> BagStats:
        """Run the plan in-process: list, skip done (``resume``), download, map, write.

        Bounded scratch (via :func:`batched_files`), one ``put_files`` call per batch, and
        resume-by-output-existence — safe to kill and re-run.
        """
        if self.fn is None:
            raise ValueError("compute() needs a map_batches(fn) stage")
        if self.out is None:
            raise ValueError("compute() needs a to_bucket(out) stage")

        todo, skipped = self._list_todo(resume=resume)
        if skipped:
            logger.info("Bag.compute: %d inputs already done, %d to do", skipped, len(todo))
        t0 = time.monotonic()
        ctx = self.setup() if self.setup is not None else None
        processed = batches = outputs = 0
        for batch in batched_files(
            self.src,
            keys=todo,
            n=self.batch_size,
            max_bytes=self.max_bytes,
            prefetch=prefetch,
            dir=dir,
            token=self.token,
        ):
            pairs = list(_call_fn(self.fn, batch, ctx, self.setup is not None))
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
