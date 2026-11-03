# bucketbag contract

What bucketbag promises. `tests/test_contract.py` checks each rule against an in-memory Hub
(`tests/fake_bucket.py`) that follows the Hub's own listing rules. A rule marked **gap** is one
today's code breaks; its test is a strict xfail until the fix lands.

## 1. Addressing

- A bucket reference is `ns/bucket[/prefix]` or `hf://buckets/ns/bucket[/prefix]`.
  **Gap:** a Hub web URL (`https://huggingface.co/buckets/ns/bucket/tree/prefix`) is not read
  as a bucket.
- A prefix is a string prefix of the key, as on the Hub. A trailing `/` limits it to one
  directory (`a/b/` lists `a/b/x`, not `a/bc/y`); without the slash, siblings match too.
  The same rule holds for `prefix=` and for a prefix inside the bucket reference.
  **Gap:** a prefix inside the bucket reference loses its trailing slash.
- Passing both a prefix inside the bucket reference and `prefix=` is not defined yet.
- Globs (`include`, `exclude`) match the whole key from the bucket root. `*` stays inside one
  path segment and `**` crosses `/`, as in `HfFileSystem.glob`. (`hf buckets sync --include`
  uses `fnmatch`, where `*` also crosses `/`.)
- Listings return full keys from the bucket root, sorted, files only. `start_after` skips
  keys up to and including it. `limit` keeps the first `limit` keys in the Hub's listing
  order, which is lexical today but not documented (issue #9).
- `put_files(pairs, "ns/bucket/prefix")` writes each key under that prefix.

## 2. Temporary files

`batched_files` deletes a batch's files when the loop moves past it. It also deletes them when
the loop body raises, when the caller stops early, and for batches downloaded ahead
(`prefetch`). `tests/test_batched_files.py` covers each `prefetch` level.

## 3. Resume (`Bag`)

- An input is done when an output key under the `to_bucket` prefix starts with the input's key
  (`pages/p1.jp2` → `pages/p1.jp2.md`). `compute()` processes only inputs that are not done;
  `compute(resume=False)` processes all of them.
- **Gap:** an output key that does not start with its input key is not reported. That input
  runs again on every re-run.
- `take(n)` runs the plan on the first `n` inputs and writes nothing. `setup()` runs once per
  `compute()` or `take()`.

## 4. Failures (`Bag.compute`)

| What fails | What happens |
|---|---|
| The map function raises | `compute()` raises that error. Earlier batches' outputs stay. |
| A download fails | `compute()` raises that error. Earlier batches' outputs stay. |
| An upload fails | `compute()` raises that error. Outputs that landed before it count as done.* |

In every row the temporary files are deleted, and a re-run processes exactly the inputs that
have no output. Hypothesis checks this over random mixes of failures (`tests/test_fault_fuzz.py`).

\* The server applies the files of an upload that succeed and reports the rest.
huggingface_hub 2.x raises `BucketBatchError` on that report. 1.x checked only the HTTP status,
so `compute()` went on, and the missing outputs were redone on the next run.

A file that is listed but gone by download time: with string keys
(`batched_files(keys=[...])`) the hub skips it with a warning, and bucketbag leaves it out of
the batch. `Bag` passes `BucketFile` objects, which the hub downloads by content hash without
an existence check; what happens then is not verified.

## 5. Arguments

- `token=` reaches every hub call.
- `max_bytes` with string keys raises `ValueError`: string keys carry no size.
- **Gap:** settings that cannot work are not refused. `n=None` with `max_bytes=None` makes one
  batch of every file, and `n=0` yields an empty batch first.
- The map function is called as `fn(items)`, or `fn(items, ctx)` when `setup=` is given.
  **Gap:** without `setup=`, bucketbag inspects the function's signature and passes `None` as a
  second argument when it sees two parameters, which breaks `fn(items, *, flag=False)`.
- **Gap:** `completed_keys` turns every shard read error into a warning, so a wrong `column=`
  returns an empty set and resume reruns everything. (The test lands with the fix.)
