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
from contextlib import closing
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar, overload

from huggingface_hub import BucketFile

from ._core import BucketRef, _check_batch_settings, _split_done
from ._hub import LoadedItem, _list_bucketfiles, batched_files, logger, put_files

__all__ = ["Bag", "BagStats"]

Output = tuple[str, bytes | str]
# fn(items) -> Iterable[Output], or fn(items, ctx) when setup= is given.
MapFn = Callable[..., Iterable[Output]]
C = TypeVar("C")
Stage = Literal["download", "map", "upload"]

# What to change when a stage is the limit (BagStats.hint).
_HINTS: dict[Stage, str] = {
    "download": (
        "Downloads are the limit: call boost() before the first download if the files are "
        "small, or raise prefetch."
    ),
    "map": (
        "The map function is the limit: raise batch_size if the model is under-filled, and "
        "time decoding separately (README: 'Decode, not transport'); decode in a process pool "
        "or at a smaller size."
    ),
    "upload": "Uploads are the limit: write fewer, larger outputs, or raise batch_size.",
}

# The shortest gap between two progress lines, in seconds (lines come after a batch).
_PROGRESS_EVERY_S = 30.0


def _shares(wait_s: float, map_s: float, upload_s: float) -> str:
    """The time split as percentages, e.g. ``download wait 5%, map 90%, upload 5%``."""
    total = (wait_s + map_s + upload_s) or 1.0
    return (
        f"download wait {wait_s / total:.0%}, map {map_s / total:.0%}, "
        f"upload {upload_s / total:.0%}"
    )


@dataclass(frozen=True)
class BagStats:
    """What a :meth:`Bag.compute` run did, and where its time went.

    ``wait_s`` is time spent waiting for the next batch (its download, plus deleting the last
    batch's files), ``map_s`` time in the map function and ``upload_s`` time writing outputs.
    ``bound_by`` names the largest of the three and ``hint`` the setting to change, so a person
    or an agent can tune the next run. The three exclude ``setup()`` and listing; ``seconds``
    includes them.
    """

    processed: int
    skipped: int
    batches: int
    outputs: int
    seconds: float
    wait_s: float = 0.0
    map_s: float = 0.0
    upload_s: float = 0.0

    @property
    def bound_by(self) -> Stage | None:
        """The stage that took the most time, or ``None`` when no batch ran."""
        if not self.batches:
            return None
        times: dict[Stage, float] = {
            "download": self.wait_s,
            "map": self.map_s,
            "upload": self.upload_s,
        }
        slowest: Stage = "download"
        for stage, seconds in times.items():
            if seconds > times[slowest]:
                slowest = stage
        return slowest

    @property
    def hint(self) -> str:
        """One sentence naming the setting to change, or ``""`` when no batch ran."""
        stage = self.bound_by
        return _HINTS[stage] if stage is not None else ""

    def __str__(self) -> str:  # print(bag.compute())
        line = (
            f"{self.processed} processed (+{self.skipped} skipped as done) in "
            f"{self.batches} batches -> {self.outputs} outputs, {self.seconds:.1f}s"
        )
        if self.bound_by is None:
            return line
        split = _shares(self.wait_s, self.map_s, self.upload_s)
        return f"{line}\ntime: {split}; bound by {self.bound_by}. {self.hint}"


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
    written = {k for k, _ in pairs}
    key = next((item.key for item in batch if item.key + done not in written), None)
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


def _log_progress(inputs: int, total: int, elapsed: float, split: str) -> None:
    """Log inputs done out of ``total``, the rate, the time left and the time split."""
    rate = inputs / elapsed if elapsed > 0 else 0.0
    left = f", about {(total - inputs) / rate / 60:.0f} min left" if rate else ""
    logger.info("%d/%d inputs, %.1f/s%s; time: %s", inputs, total, rate, left, split)


def _run(
    apply: Callable[[list[LoadedItem]], Iterable[Output]],
    batches: Iterable[list[LoadedItem]],
    sink: Callable[[list[Output]], None],
    done: str | None,
    total: int,
) -> BagStats:
    """Map every batch and hand its outputs to ``sink``, timing each stage.

    Warns once per run about an input whose ``key + done`` output is missing, and logs a
    progress line after a batch, at most every ``_PROGRESS_EVERY_S`` seconds.
    """
    inputs = count = outputs = 0
    wait_s = map_s = upload_s = 0.0
    warned = False
    started = last_log = waiting_since = time.monotonic()
    for batch in batches:  # each step waits for the next download
        t1 = time.monotonic()
        pairs = list(apply(batch))
        t2 = time.monotonic()
        sink(pairs)
        t3 = time.monotonic()
        wait_s += t1 - waiting_since
        map_s += t2 - t1
        upload_s += t3 - t2
        inputs += len(batch)
        count += 1
        outputs += len(pairs)
        if done is not None and not warned:
            warned = _warn_if_not_done(batch, pairs, done)
        if t3 - last_log >= _PROGRESS_EVERY_S:
            _log_progress(inputs, total, t3 - started, _shares(wait_s, map_s, upload_s))
            last_log = t3
        waiting_since = time.monotonic()
    return BagStats(
        processed=inputs,
        skipped=0,
        batches=count,
        outputs=outputs,
        seconds=time.monotonic() - started,
        wait_s=wait_s,
        map_s=map_s,
        upload_s=upload_s,
    )


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
        # Closed here, not left to garbage collection: a caller that keeps a raised exception
        # also keeps this frame, and with it the generator and its temp files.
        with closing(batches):
            _run(apply, batches, outputs.extend, self.done, total=len(files))
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
        logger.info(
            "Bag.compute: %d to do, %d already done; batch_size=%s, max_bytes=%s, prefetch=%d",
            len(todo),
            skipped,
            self.batch_size,
            self.max_bytes,
            prefetch,
        )
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

        with closing(batches):  # see take(): a kept exception must not keep the temp files
            stats = _run(apply, batches, upload, done, total=len(todo))
        stats = replace(stats, skipped=skipped, seconds=time.monotonic() - t0)
        logger.info("Bag.compute: %s", stats)
        return stats
