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
- Globs (`include`, `exclude`) match the whole key from the bucket root. `*` stays inside one
  path segment and `**` crosses `/`, as in `HfFileSystem.glob`. (`hf buckets sync --include`
  uses `fnmatch`, where `*` also crosses `/`.)
- Listings return full keys from the bucket root, sorted.
- `put_files(pairs, "ns/bucket/prefix")` writes each key under that prefix.

## 2. Temporary files

`batched_files` deletes a batch's files when the loop moves past it. It also deletes them when
the loop body raises, when the caller stops early, and for batches downloaded ahead
(`prefetch`). `tests/test_batched_files.py` covers each `prefetch` level.

## 3. Resume (`Bag`)

- An input is done when an output key under the `to_bucket` prefix starts with the input's key
  (`pages/p1.jp2` → `pages/p1.jp2.md`). `compute()` processes only inputs that are not done.
- **Gap:** an output key that does not start with its input key is not reported. That input
  runs again on every re-run.

## 4. Failures (`Bag.compute`)

| What fails | What happens |
|---|---|
| The map function raises | `compute()` raises that error. Earlier batches' outputs stay. |
| A download fails | `compute()` raises that error. Earlier batches' outputs stay. |
| An upload fails | `compute()` raises that error. Outputs that landed before it count as done.* |
| A listed file is gone at download time | The input is skipped with a warning and stays not done. |

In every row the temporary files are deleted, and a re-run processes exactly the inputs that
have no output. Hypothesis checks this over random mixes of failures (`tests/test_fault_fuzz.py`).

\* huggingface_hub 2.x sends a large upload in chunks, applies every chunk that succeeds, then
raises `BucketBatchError`, so part of a batch can land. On hub 1.x a partial failure returned
normally: `compute()` went on, and the missing outputs were redone on the next run.

**Gap:** `BagStats` does not count skipped missing files, so `listed = processed + skipped +
missing` cannot be checked.

## 5. Arguments

- **Gap:** settings that cannot work are not refused. `n=None` with `max_bytes=None` makes one
  batch of every file, and `n=0` yields an empty batch first.
- The map function is called as `fn(items)`, or `fn(items, ctx)` when `setup=` is given.
  **Gap:** without `setup=`, bucketbag inspects the function's signature and passes `None` as a
  second argument when it sees two parameters, which breaks `fn(items, *, flag=False)`.
- **Gap:** `completed_keys` turns every shard read error into a warning, so a wrong `column=`
  returns an empty set and resume reruns everything. (The test lands with the fix.)
