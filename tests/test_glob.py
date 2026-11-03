"""Tests for the glob rules (fsspec's, as in ``HfFileSystem.glob``) and ``_glob_prefix``."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bucketbag._core import _glob_prefix, _glob_to_re

pytestmark = pytest.mark.internals

# A "literal" alphabet: any chars except the glob metacharacters and surrogate/whitespace noise.
_LITERAL = st.characters(blacklist_categories=("Cs",), blacklist_characters="*?[ \t\n/")


# --- _glob_to_re: concrete semantics ---------------------------------------
@pytest.mark.parametrize(
    "pattern, key, matches",
    [
        # `*` stays within a path segment (does NOT cross `/`).
        ("*.jp2", "a.jp2", True),
        ("*.jp2", "dir/a.jp2", False),
        ("*.jp2", "a.jp2.bak", False),
        # `**/` matches zero or more directories.
        ("**/*.jp2", "a.jp2", True),
        ("**/*.jp2", "dir/a.jp2", True),
        ("**/*.jp2", "a/b/c/a.jp2", True),
        # `**` as a whole segment matches across `/`.
        ("a/**", "a/b/c/z", True),
        # `[...]` is a character class.
        ("[ab].txt", "a.txt", True),
        ("[ab].txt", "c.txt", False),
        # `?` matches exactly one non-`/` char.
        ("?.txt", "a.txt", True),
        ("?.txt", "ab.txt", False),
        ("?.txt", "/.txt", False),
        # literal chars are escaped: `.` is not "any char".
        ("a.txt", "a.txt", True),
        ("a.txt", "abtxt", False),
        ("a.txt", "aXtxt", False),
    ],
)
def test_glob_to_re_cases(pattern: str, key: str, matches: bool) -> None:
    assert bool(_glob_to_re(pattern).match(key)) is matches


@pytest.mark.parametrize("pattern", ["a**z", "**.jp2", "pages/**.md"])
def test_double_star_inside_a_name_is_refused(pattern: str) -> None:
    with pytest.raises(ValueError, match="own path segment"):
        _glob_to_re(pattern)


def test_glob_to_re_cache_returns_same_object() -> None:
    a = _glob_to_re("*.bin")
    b = _glob_to_re("*.bin")
    assert a is b  # cached


# --- _glob_prefix -----------------------------------------------------------
@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("images/**/*.jp2", "images/"),
        ("a/b/c.txt", "a/b/"),
        ("*.jp2", ""),
        ("?.txt", ""),
        ("x", ""),
        ("[ab].txt", ""),
        ("data/*.csv", "data/"),
    ],
)
def test_glob_prefix_cases(pattern: str, expected: str) -> None:
    assert _glob_prefix(pattern) == expected


# --- properties -------------------------------------------------------------
@given(st.text(min_size=1, max_size=16, alphabet=st.characters(blacklist_categories=("Cs",))))
def test_glob_prefix_never_contains_wildcards(pattern: str) -> None:
    prefix = _glob_prefix(pattern)
    assert not set(prefix) & {"*", "?", "["}


@given(st.text(min_size=1, max_size=12, alphabet=_LITERAL))
def test_literal_pattern_matches_only_itself(literal: str) -> None:
    rx = _glob_to_re(literal)
    assert rx.match(literal)
    assert not rx.match(literal + "x")


# Path-like literals: 1-4 wildcard-free segments joined by "/", so non-empty prefixes are
# actually generated (a flat _LITERAL alphabet excludes "/", so it could only ever yield "").
_LITERAL_SEG = st.text(min_size=1, max_size=6, alphabet=_LITERAL)
_LITERAL_PATH = st.lists(_LITERAL_SEG, min_size=1, max_size=4).map("/".join)


@given(_LITERAL_PATH)
def test_glob_prefix_is_everything_up_to_last_slash(path: str) -> None:
    # With no wildcard, the prefix is the literal head up to (and including) the final "/".
    prefix = _glob_prefix(path)
    assert path.startswith(prefix)
    slash = path.rfind("/")
    assert prefix == (path[: slash + 1] if slash >= 0 else "")
