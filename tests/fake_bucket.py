"""An in-memory stand-in for the three ``HfApi`` bucket methods that bucketbag calls.

One store serves listings, downloads and uploads, so a test can write with ``put_files`` and
resume from what was written, the same way a real run does.

Listing rules follow the Hub, measured on 2026-11-03 against ``biglam/britannica`` with
huggingface_hub 1.21.0 and 2.4.0: a recursive listing returns every file whose key starts with
the prefix string. A trailing ``/`` therefore scopes the listing to one directory, and a prefix
without one also matches siblings (``a/b`` matches ``a/bc/y``). Recursive listings return files
only, in lexical key order.

Each call is first bound against the real ``HfApi`` signature of the installed
huggingface_hub. If bucketbag calls a method with arguments the real method rejects, the test
fails with ``TypeError`` instead of passing against the fake.
"""

from __future__ import annotations

import inspect
import threading
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pytest
from huggingface_hub import BucketFile, BucketFolder, HfApi
from huggingface_hub.errors import EntryNotFoundError

METHODS = ("list_bucket_tree", "download_bucket_files", "batch_bucket_files")

# Read before any test patches HfApi, so these are always the installed library's signatures.
REAL_SIGNATURES = {name: inspect.signature(getattr(HfApi, name)) for name in METHODS}


class FaultInjected(RuntimeError):
    """Raised by a FakeBucket call that the test asked to fail."""


@dataclass
class FakeBucket:
    """Buckets as ``{bucket_id: {key: bytes}}``, plus switches for the failures tests need.

    Failure switches:
        fail_download_on / fail_upload_on: 1-based call numbers, counted per method. That call
            raises ``FaultInjected`` (for downloads, ``download_error``) before it reads or
            writes anything.
        partial_upload_on: that upload call stores the first half of its files, then raises
            ``FaultInjected``. The server applies the files of a batch that succeed and reports
            the rest; huggingface_hub 2.x raises ``BucketBatchError`` on that report.
        download_seconds / upload_seconds: each download or upload call sleeps this long, so
            a test can interrupt a wait or make that stage the slow one.
        missing: keys that a listing returns but that are gone by download time. As in the
            hub, a string key is skipped with a warning. A ``BucketFile`` skips the hub's
            existence check (it downloads by content hash), so what the real hub does then is
            unverified; the fake raises the hub's ``EntryNotFoundError``.
        overmatch_after_first_page: replay the Hub pagination bug fixed in September 2026.
            After a full first page the trailing slash was lost, so siblings leaked in.
        listing_order: ``"reversed"`` returns keys in reverse order. The Hub does not document
            its listing order (bucketbag issue #9), so bucketbag sorts what it receives.
        folders: recursive listings also yield a ``BucketFolder`` per directory. The Hub does
            not do this today; the switch exercises bucketbag's folder skip.
    """

    store: dict[str, dict[str, bytes]] = field(default_factory=dict)
    page_size: int = 1000
    overmatch_after_first_page: bool = False
    listing_order: Literal["sorted", "reversed"] = "sorted"
    folders: bool = False
    missing: set[str] = field(default_factory=set)
    fail_download_on: set[int] = field(default_factory=set)
    fail_upload_on: set[int] = field(default_factory=set)
    partial_upload_on: set[int] = field(default_factory=set)
    download_seconds: float = 0.0
    upload_seconds: float = 0.0
    download_error: type[BaseException] = FaultInjected
    dirs_at_download: list[int] = field(default_factory=list)
    calls: Counter[str] = field(default_factory=Counter)
    list_prefixes: list[str | None] = field(default_factory=list)
    uploads: list[tuple[str, list[str]]] = field(default_factory=list)
    tokens: list[tuple[str, object]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _count(self, method: str) -> int:
        """Count a call and return its 1-based number. Prefetch threads call concurrently."""
        with self._lock:
            self.calls[method] += 1
            return self.calls[method]

    # -- set up and inspect state from a test ---------------------------------------------
    def put(self, bucket_id: str, key: str, data: bytes = b"xxxx") -> BucketFile:
        """Store one file and return the ``BucketFile`` a listing would return for it."""
        self.store.setdefault(bucket_id, {})[key] = data
        return self.entry(bucket_id, key)

    def seed(self, bucket_id: str, keys: Iterable[str], data: bytes = b"xxxx") -> list[BucketFile]:
        """Store several files with the same content; return their ``BucketFile`` entries."""
        return [self.put(bucket_id, key, data) for key in keys]

    def entry(self, bucket_id: str, key: str) -> BucketFile:
        size = len(self.store[bucket_id][key])
        return BucketFile(type="file", path=key, size=size, xetHash="")

    def keys(self, bucket_id: str) -> list[str]:
        return sorted(self.store.get(bucket_id, {}))

    def get(self, bucket_id: str, key: str) -> bytes:
        return self.store[bucket_id][key]

    def uploaded_keys(self, bucket_id: str) -> list[str]:
        """Every key written to ``bucket_id``, one entry per write (repeats show double writes)."""
        return [key for bid, keys in self.uploads if bid == bucket_id for key in keys]

    # -- the HfApi methods -----------------------------------------------------------------
    def list_bucket_tree(
        self,
        bucket_id: str,
        prefix: str | None = None,
        *,
        recursive: bool | None = None,
        token: str | bool | None = None,
    ) -> Iterator[BucketFile | BucketFolder]:
        self._count("list")
        self.list_prefixes.append(prefix)
        if not recursive:
            raise NotImplementedError("FakeBucket models recursive listings only")
        keys = self._matching_keys(bucket_id, prefix or "")
        if self.listing_order == "reversed":
            keys.reverse()
        entries: list[BucketFile | BucketFolder] = []
        if self.folders:
            directories = sorted({str(PurePosixPath(key).parent) for key in keys} - {"."})
            entries += [BucketFolder(type="directory", path=d) for d in directories]
        entries += [self.entry(bucket_id, key) for key in keys]
        return iter(entries)  # the hub returns a one-pass iterator, not a list

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str | BucketFile, str | Path]],
        *,
        raise_on_missing_files: bool = False,
        token: str | bool | None = None,
    ) -> None:
        number = self._count("download")
        if files:
            # Batch dirs on disk as this download starts, its own included (the high-water mark).
            batch_dir = Path(files[0][1]).parent
            self.dirs_at_download.append(len(list(batch_dir.parent.glob("bb-*"))))
        if number in self.fail_download_on:
            raise self.download_error(f"download call #{number} failed")
        data = self.store.get(bucket_id, {})
        for remote, local in files:
            key = remote.path if isinstance(remote, BucketFile) else remote
            if key in self.missing or key not in data:
                if raise_on_missing_files or isinstance(remote, BucketFile):
                    raise EntryNotFoundError(f"File '{key}' not found in bucket '{bucket_id}'.")
                warnings.warn(
                    f"File '{key}' not found in bucket '{bucket_id}'. Skipping.", stacklevel=2
                )
                continue
            Path(local).write_bytes(data[key])
        if self.download_seconds:
            time.sleep(self.download_seconds)

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[str | Path | bytes, str]] | None = None,
        copy: list[tuple[str, str, str, str]] | None = None,
        delete: list[str] | None = None,
        token: str | bool | None = None,
    ) -> None:
        number = self._count("upload")
        if number in self.fail_upload_on:
            raise FaultInjected(f"upload call #{number} failed")
        if copy or delete:
            raise NotImplementedError("FakeBucket models add only")
        staged = [(dest, _as_bytes(content)) for content, dest in add or []]
        partial = number in self.partial_upload_on
        if partial:
            staged = staged[: len(staged) // 2]
        self.store.setdefault(bucket_id, {}).update(staged)
        self.uploads.append((bucket_id, [dest for dest, _ in staged]))
        if self.upload_seconds:
            time.sleep(self.upload_seconds)
        if partial:
            raise FaultInjected(f"upload call #{number} stored {len(staged)} file(s), then failed")

    # -- wiring ----------------------------------------------------------------------------
    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeBucket:
        """Route the three ``HfApi`` methods to this store for the rest of the test."""
        for name in METHODS:
            monkeypatch.setattr(HfApi, name, self._checked(name))
        return self

    def _checked(self, name: str) -> Callable[..., Any]:
        real = REAL_SIGNATURES[name]
        fake = getattr(self, name)

        def call(api: HfApi, *args: Any, **kwargs: Any) -> Any:
            bound = real.bind(api, *args, **kwargs)  # TypeError if the real method would reject it
            # The token the call would send: its own token= argument, else the HfApi's.
            self.tokens.append((name, bound.arguments.get("token") or api.token))
            return fake(*args, **kwargs)

        return call

    def _matching_keys(self, bucket_id: str, prefix: str) -> list[str]:
        all_keys = sorted(self.store.get(bucket_id, {}))
        matching = [key for key in all_keys if key.startswith(prefix)]
        first_page = matching[: self.page_size]
        if not self.overmatch_after_first_page or len(first_page) < self.page_size:
            return matching
        # The old bug: pages after the first matched the prefix without its trailing slash.
        slashless = prefix.rstrip("/")
        later = [key for key in all_keys if key.startswith(slashless) and key > first_page[-1]]
        return first_page + later


def _as_bytes(content: str | Path | bytes) -> bytes:
    if isinstance(content, bytes):
        return content
    return Path(content).read_bytes()
