"""``Bag`` — a lazy, dask.bag-style plan over bucket files (local execution).

The chain is a *plan*, not work: nothing lists or downloads until :meth:`Bag.compute` /
:meth:`Bag.take`. One shape, three verbs::

    from bucketbag import Bag

    bag = (Bag.from_bucket(SRC, include="**/*.jp2")
             .map_batches(ocr_batch, batch_size=32, setup=load_model)
             .to_bucket(OUT, done=".md"))
    bag.take(3)        # dev: run the fn on the first few items, return outputs, write nothing
    bag.compute()      # run: list -> skip done -> batched download -> fn -> put_files, bounded

The map function receives a ``list[LoadedItem]`` (plus the ``setup()`` result if given) and
returns an iterable of ``(key, content)`` pairs — content is ``bytes`` or ``str``. Outputs land
under the ``to_bucket`` prefix in one API call per batch.

**Resume:** ``to_bucket(out, done=".md")`` names the output that marks an input done. An input
``k`` is done when ``k + ".md"`` exists under ``out``, so ``compute()`` (``resume=True`` by
default) skips it on a re-run. The lookup is exact: no other key can mark ``k`` done.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any, TypeVar, overload

from huggingface_hub import BucketFile

from ._core import BucketRef, _check_batch_settings, _first_not_done, _split_done
from ._hub import LoadedItem, _list_bucketfiles, batched_files, logger, put_files

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


def _warn_if_not_done(batch: list[LoadedItem], pairs: list[Output], done: str) -> bool:
    """Log a warning if an input got no ``key + done`` output; return whether it did."""
    key = _first_not_done([item.key for item in batch], [k for k, _ in pairs], done)
    if key is None:
        return False
    logger.warning(
        "Input %r produced no %r output, so resume cannot mark it done and it runs again on "
        "every re-run. Write <input key>%s for each input, or change to_bucket(done=...).",
        key,
        key + done,
        done,
    )
    return True


def _run(
    apply: Callable[[list[LoadedItem]], Iterable[Output]],
    batches: Iterable[list[LoadedItem]],
    sink: Callable[[list[Output]], object],
    done: str | None,
) -> tuple[int, int, int]:
    """Map every batch and hand its outputs to ``sink``; return (inputs, batches, outputs).

    Warns once per run about an input whose ``key + done`` output is missing.
    """
    inputs = count = outputs = 0
    warned = False
    for batch in batches:
        pairs = list(apply(batch))
        if done is not None and not warned:
            warned = _warn_if_not_done(batch, pairs, done)
        sink(pairs)
        inputs += len(batch)
        count += 1
        outputs += len(pairs)
    return inputs, count, outputs


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
    done: str | None = None

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
        """Start a plan over ``bucket`` (a reference as in :func:`~bucketbag.iter_keys`)."""
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

    def to_bucket(self, out: str, *, done: str | None = None) -> Bag:
        """Set where the fn's ``(key, content)`` pairs land, and which output marks an input done.

        ``done`` is a suffix: an input ``k`` is done when ``k + done`` exists under ``out``
        (``done=".md"`` for ``pages/p1.jp2`` -> ``pages/p1.jp2.md``). ``compute(resume=True)``
        needs it.
        """
        return replace(self, out=out, done=done)

    # -- execution --------------------------------------------------------------
    def _list_inputs(self, *, limit: int | None) -> list[BucketFile]:
        """List source inputs as ``BucketFile`` objects (sizes -> max_bytes + fast download)."""
        return _list_bucketfiles(
            BucketRef.parse(self.src),
            include=self.include,
            exclude=self.exclude,
            start_after=None,
            limit=limit,
            token=self.token,
        )

    def _list_todo(self, out: str, done: str) -> tuple[list[BucketFile], int]:
        files = self._list_inputs(limit=self.limit)
        out_ref = BucketRef.parse(out)
        existing = _list_bucketfiles(
            out_ref, include=None, exclude=None, start_after=None, limit=None, token=self.token
        )
        return _split_done(files, (f.path for f in existing), out_ref, done)

    def take(self, n: int, *, dir: str | os.PathLike[str] | None = None) -> list[Output]:
        """Dev peek: run the plan on the first ``n`` items and return the outputs. No writes."""
        fn = self.fn
        if fn is None:
            raise ValueError("take() needs a map_batches(fn) stage")
        files = self._list_inputs(limit=n)
        apply = _with_context(fn, self.setup)
        outputs: list[Output] = []
        batches = batched_files(
            self.src,
            keys=files,
            n=self.batch_size,
            max_bytes=self.max_bytes,
            prefetch=0,
            dir=dir,
            token=self.token,
        )
        _run(apply, batches, outputs.extend, self.done)
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
        resume by the ``to_bucket(done=...)`` output — safe to kill and re-run.
        """
        fn, out, done = self.fn, self.out, self.done
        if fn is None:
            raise ValueError("compute() needs a map_batches(fn) stage")
        if out is None:
            raise ValueError("compute() needs a to_bucket(out) stage")
        if resume and done is None:
            raise ValueError(
                "compute(resume=True) needs to know which output marks an input done: use "
                "to_bucket(out, done='.md') when each input k writes k + '.md', or "
                "compute(resume=False) to process every input."
            )
        _check_batch_settings(self.batch_size, self.max_bytes, prefetch)

        if resume and done is not None:
            todo, skipped = self._list_todo(out, done)
        else:
            todo, skipped = self._list_inputs(limit=self.limit), 0
        if skipped:
            logger.info("Bag.compute: %d inputs already done, %d to do", skipped, len(todo))
        t0 = time.monotonic()
        apply = _with_context(fn, self.setup)
        batches = batched_files(
            self.src,
            keys=todo,
            n=self.batch_size,
            max_bytes=self.max_bytes,
            prefetch=prefetch,
            dir=dir,
            token=self.token,
        )

        def upload(pairs: list[Output]) -> None:
            put_files(pairs, out, token=self.token)

        processed, count, outputs = _run(apply, batches, upload, done)
        return BagStats(
            processed=processed,
            skipped=skipped,
            batches=count,
            outputs=outputs,
            seconds=time.monotonic() - t0,
        )
