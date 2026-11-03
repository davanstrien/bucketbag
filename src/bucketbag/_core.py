"""The pure core: addressing, listing filters, batching and resume decisions.

Nothing here calls the Hub or touches disk. ``_hub.py`` does the I/O and asks these functions
what to do, so each rule in CONTRACT.md can be tested on its own.
"""

from __future__ import annotations

import functools
import math
import numbers
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from fsspec.utils import glob_translate
from huggingface_hub import BucketFile, parse_hf_uri

_HF_SCHEME = "hf://"


# ---------------------------------------------------------------------------
# Addressing (CONTRACT §1)
# ---------------------------------------------------------------------------
def _join(prefix: str, rest: str) -> str:
    """Join two key prefixes as directories: ``"a/b" + "k"`` and ``"a/b/" + "k"`` give ``a/b/k``."""
    if not prefix:
        return rest
    if not rest:
        return prefix
    return prefix.rstrip("/") + "/" + rest


@dataclass(frozen=True)
class BucketRef:
    """A parsed bucket reference: the bucket id and a key prefix.

    The prefix is a string prefix of keys, as on the Hub: a trailing ``/`` limits it to one
    directory (``a/b/`` lists ``a/b/x``, not ``a/bc/y``).
    """

    bucket_id: str
    prefix: str = ""

    @classmethod
    def parse(cls, ref: str) -> BucketRef:
        """Read ``ns/bucket[/prefix]``, ``hf://buckets/ns/bucket[/prefix]`` or a Hub web URL."""
        if ref.startswith(_HF_SCHEME) or "://" in ref:
            return cls._parse_uri(ref)
        parts = ref.lstrip("/").split("/", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Bucket reference must be 'ns/bucket' (optionally with a prefix), got: {ref!r}"
            )
        prefix = parts[2] if len(parts) == 3 else ""
        return cls(f"{parts[0]}/{parts[1]}", prefix)

    @classmethod
    def _parse_uri(cls, ref: str) -> BucketRef:
        parsed = parse_hf_uri(ref)
        if parsed.type != "bucket":
            raise ValueError(f"Not a bucket reference: {ref!r} points at a {parsed.type}.")
        # parse_hf_uri drops a trailing "/", so read it back: from the URI itself, or from a web
        # URL's path, where a query string must neither add nor remove it.
        path = ref if ref.startswith(_HF_SCHEME) else urlsplit(ref).path
        slash = "/" if parsed.path_in_repo and path.endswith("/") else ""
        return cls(parsed.id, parsed.path_in_repo + slash)

    def under(self, prefix: str | None) -> BucketRef:
        """This reference with ``prefix`` joined below its own prefix, as directories."""
        if not prefix:
            return self
        return BucketRef(self.bucket_id, _join(self.prefix, prefix))

    def key_for(self, key: str) -> str:
        """The full bucket key where ``put_files`` writes ``key``."""
        return _join(self.prefix, key)


# ---------------------------------------------------------------------------
# Listing filters (CONTRACT §1)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def _glob_to_re(pattern: str) -> re.Pattern[str]:
    """Compile a glob with fsspec's rules, the ones ``HfFileSystem.glob`` uses.

    ``*`` stays inside one path segment, ``**`` as a whole segment crosses ``/``, ``?`` is one
    character and ``[...]`` a character class.
    """
    try:
        return re.compile(glob_translate(pattern))
    except ValueError as exc:
        raise ValueError(
            f"Invalid glob {pattern!r} ({exc}). Write '**' as its own path segment, "
            "e.g. '**/*.jp2'."
        ) from exc


def _glob_prefix(pattern: str) -> str:
    """The literal leading directory of a glob, used as a cheap server-side list prefix.

    ``"images/**/*.jp2" -> "images/"``; ``"a/b/c.txt" -> "a/b/"``; ``"*.jp2" -> ""``.
    """
    head = pattern
    for i, ch in enumerate(pattern):
        if ch in "*?[":
            head = pattern[:i]
            break
    slash = head.rfind("/")
    return head[: slash + 1] if slash >= 0 else ""


def _keeps(
    key: str,
    prefix: str,
    include: re.Pattern[str] | None,
    exclude: re.Pattern[str] | None,
    start_after: str | None,
) -> bool:
    """Whether a listing keeps ``key``.

    The prefix check also guards against a listing that returns keys outside the prefix.
    """
    if not key.startswith(prefix):
        return False
    if include is not None and include.match(key) is None:
        return False
    if exclude is not None and exclude.match(key) is not None:
        return False
    return start_after is None or key > start_after


# ---------------------------------------------------------------------------
# Batches (CONTRACT §5)
# ---------------------------------------------------------------------------
def _is_count(value: object) -> bool:
    """A whole number that is not a bool (Python counts ``True`` as 1)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_byte_cap(value: object) -> bool:
    """A finite real number that is not a bool. ``4e9`` is fine.

    nan and inf compare False with every running total, so ``_pack`` would never split by
    bytes and one batch would hold every file.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(value)


def _check_batch_settings(n: int | None, max_bytes: float | None, prefetch: int = 0) -> None:
    """Refuse batch settings that cannot work, before anything is listed or downloaded."""
    if n is not None and not _is_count(n):
        raise TypeError(f"n must be a whole number or None, got {n!r}.")
    if max_bytes is not None and not _is_byte_cap(max_bytes):
        raise TypeError(f"max_bytes must be a finite number or None, got {max_bytes!r}.")
    if not _is_count(prefetch):
        raise TypeError(f"prefetch must be a whole number, got {prefetch!r}.")
    if n is None and max_bytes is None:
        raise ValueError(
            "Batches need a bound: set the file count (n / batch_size), max_bytes, or both. "
            "With neither, one batch would hold every file."
        )
    if n is not None and n < 1:
        raise ValueError(f"The file count per batch must be at least 1, got {n}.")
    if max_bytes is not None and max_bytes < 1:
        raise ValueError(f"max_bytes must be at least 1, got {max_bytes}.")
    if prefetch < 0:
        raise ValueError(f"prefetch must be 0 or more, got {prefetch}.")


def _pack(
    remotes: Sequence[BucketFile | str], n: int | None, max_bytes: float | None
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


# ---------------------------------------------------------------------------
# Resume (CONTRACT §3)
# ---------------------------------------------------------------------------
def _split_done(
    files: Sequence[BucketFile], out_keys: Iterable[str], out: BucketRef, done: str
) -> tuple[list[BucketFile], int]:
    """The inputs still to do, and how many are done.

    An input ``k`` is done when the output ``k + done`` exists under ``out``.
    """
    existing = set(out_keys)
    todo = [f for f in files if out.key_for(f.path + done) not in existing]
    return todo, len(files) - len(todo)


def _first_not_done(input_keys: Iterable[str], output_keys: Iterable[str], done: str) -> str | None:
    """The first input key with no ``key + done`` among its batch's outputs, else ``None``."""
    written = set(output_keys)
    for key in input_keys:
        if key + done not in written:
            return key
    return None
