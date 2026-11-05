"""xet's process-wide settings: what importing bucketbag sets, what ``boost()`` changes, and
the one log line that reports them (CONTRACT §7).

Import-time behaviour is checked in a fresh interpreter, because the settings are applied once,
when bucketbag is first imported.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fake_bucket import FakeBucket

from bucketbag import _xet_env, batched_files

HP = "HF_XET_HIGH_PERFORMANCE"
CAP = "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"
OPT_OUT = "BUCKETBAG_NO_XET_TUNE"


def settings_after(code: str, **env: str) -> dict[str, str | None]:
    """Run ``code`` after ``import bucketbag`` in a fresh interpreter; return the two settings."""
    clean = {k: v for k, v in os.environ.items() if k not in (HP, CAP, OPT_OUT)}
    clean.update(env)
    script = (
        "import json, os, bucketbag\n"
        f"{code}\n"
        f"print(json.dumps({{'hp': os.environ.get({HP!r}), 'cap': os.environ.get({CAP!r})}}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], env=clean, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_sets_high_performance_and_32_concurrent_files() -> None:
    assert settings_after("") == {"hp": "1", "cap": "32"}


def test_import_keeps_values_the_caller_exported() -> None:
    assert settings_after("", **{CAP: "8", HP: "0"}) == {"hp": "0", "cap": "8"}


def test_opt_out_keeps_xet_defaults() -> None:
    assert settings_after("", **{OPT_OUT: "1"}) == {"hp": None, "cap": None}


def test_boost_replaces_the_import_default() -> None:
    assert settings_after("bucketbag.boost(file_concurrency=64)") == {"hp": "1", "cap": "64"}


def test_boost_never_replaces_an_exported_value() -> None:
    assert settings_after("bucketbag.boost(file_concurrency=64)", **{CAP: "16"})["cap"] == "16"


def test_boost_after_opt_out_sets_what_it_is_asked() -> None:
    assert settings_after("bucketbag.boost(file_concurrency=48)", **{OPT_OUT: "1"}) == {
        "hp": "1",
        "cap": "48",
    }


@pytest.mark.internals
def test_batched_files_logs_the_settings_once_per_process(
    bucket: FakeBucket,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_xet_env, "_logged", False)
    keys = bucket.seed("ns/b", ["a.bin", "b.bin"])
    with caplog.at_level(logging.INFO, logger="bucketbag"):
        for _ in range(2):
            for _batch in batched_files("ns/b", keys=keys, n=1, prefetch=0, dir=tmp_path):
                pass
    lines = [r.getMessage() for r in caplog.records if "xet settings" in r.getMessage()]
    assert len(lines) == 1
    assert "every Hugging Face download in this process" in lines[0]
    assert CAP in lines[0]


@pytest.mark.internals
def test_the_log_line_names_who_set_each_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_xet_env, "_set_by_bucketbag", {CAP: "32"})
    monkeypatch.setenv(CAP, "32")
    monkeypatch.setenv(HP, "0")
    line = _xet_env.describe()
    assert f"{CAP}=32 (set by bucketbag)" in line
    assert f"{HP}=0 (from your environment)" in line
