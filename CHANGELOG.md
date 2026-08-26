# Changelog

## 0.3.1 — 2026-08-26

- Fix: `prefix` over-matched sibling directories in `iter_keys`, `batched_files` and
  `completed_keys`. `prefix="a/b/"` returned `a/bc/y` too, because the trailing slash was
  dropped before the Hub call and the Hub's `list_bucket_tree(prefix=)` is a plain string
  prefix either way. Keys are now filtered client-side with the caller's original prefix:
  a trailing `/` means "this directory"; no slash keeps string-prefix semantics.

## 0.3.0

- `max_bytes` raises (not warns) when it cannot be honored (string keys carry no size).
- See git tags `v0.3.0` / `v0.2.0` for earlier release notes.
