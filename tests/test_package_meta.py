"""The version in the code matches the version the package was built with."""

from __future__ import annotations

import importlib.metadata

import bucketbag


def test_version_matches_the_package_metadata() -> None:
    assert bucketbag.__version__ == importlib.metadata.version("bucketbag")
