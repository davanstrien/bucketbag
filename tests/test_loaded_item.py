"""Tests for the ``LoadedItem`` dataclass (decoders + lazy byte caching).

These write real files to ``tmp_path``; no network and no ``batched_files`` involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bucketbag import LoadedItem


def test_bytes_reads_and_caches(tmp_path: Path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"payload")
    it = LoadedItem(key="k", path=f)
    assert it._bytes is None
    assert it.bytes == b"payload"
    assert it._bytes == b"payload"  # cached now
    # second read hits the cache, not disk (delete the file to prove it)
    f.unlink()
    assert it.bytes == b"payload"


def test_text_decodes(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("héllo, world", encoding="utf-8")
    it = LoadedItem(key="k", path=f)
    assert it.text() == "héllo, world"


def test_text_custom_encoding(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes("café".encode("latin-1"))
    assert LoadedItem(key="k", path=f).text(encoding="latin-1") == "café"


def test_json_parses(tmp_path: Path):
    f = tmp_path / "x.json"
    obj = {"a": 1, "b": [2, 3], "c": {"nested": True}}
    f.write_text(json.dumps(obj))
    assert LoadedItem(key="k", path=f).json() == obj


def test_json_invalid_raises(tmp_path: Path):
    f = tmp_path / "x.json"
    f.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        LoadedItem(key="k", path=f).json()


def test_image_opens_pil(tmp_path: Path):
    from PIL import Image

    f = tmp_path / "x.png"
    Image.new("RGB", (2, 2), (10, 20, 30)).save(f)
    img = LoadedItem(key="k", path=f).image
    assert img.size == (2, 2)
    assert img.getpixel((0, 0)) == (10, 20, 30)
