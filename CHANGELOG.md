# Changelog

## 0.3.1 — 2026-08-26

- Fix: `prefix` over-matched sibling directories in `iter_keys`, `batched_files` and
  `completed_keys`. `prefix="a/b/"` returned `a/bc/y` too. Two causes: bucketbag stripped the
  trailing slash before calling the Hub, and the Hub's pagination `Link: next` URL carries a
  literal `/` that a 302 removes, so pages after the first of a recursive listing match on the
  slash-less string prefix. The slash is now sent verbatim and keys are re-filtered
  client-side with the caller's original prefix: a trailing `/` means "this directory"; no
  slash keeps string-prefix semantics.

## 0.3.0

- `max_bytes` raises (not warns) when it cannot be honored (string keys carry no size).
- See git tags `v0.3.0` / `v0.2.0` for earlier release notes.
