## Summary

Remote CSV inputs are fully downloaded into a temporary file before
`scan_csv`, `read_csv`, or `read_csv_chunked` applies its sampling/row limit.
This makes `sample_size`, `nrows`, and chunking ineffective at limiting
network transfer or temporary-disk usage.

This also contradicts `docs/remote.md`, which describes `scan_csv` as
inspecting a schema "without downloading the full file body" and
`read_csv_chunked` as streaming a remote CSV.

## Environment

- Arnio: current `main` at `78d5ef1e5c7b3af48913a97ae1a02382139532f2`
- Python: 3.13
- OS: Windows
- Input: mocked HTTPS response emitting valid CSV incrementally

## Steps to reproduce

Use a response object that counts bytes returned by `read()` and patch
`arnio.io.urllib.request.urlopen` to return it. The response below represents
a 2 MiB valid one-column CSV delivered in 64 chunks.

```python
from unittest.mock import patch
import arnio as ar

CHUNK_SIZE = 32 * 1024
CHUNK_COUNT = 64

class CountingResponse:
    def __init__(self):
        first_rows = (CHUNK_SIZE - len(b"value\n")) // 2
        self.chunks = [b"value\n" + b"1\n" * first_rows]
        self.chunks += [b"1\n" * (CHUNK_SIZE // 2)] * (CHUNK_COUNT - 1)
        self.bytes_read = 0
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size=-1):
        self.read_calls += 1
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        self.bytes_read += len(chunk)
        return chunk

response = CountingResponse()
with patch("arnio.io.urllib.request.urlopen", return_value=response):
    print(ar.scan_csv("https://example.invalid/large.csv", sample_size=1))
print(response.bytes_read, response.read_calls)
```

Repeat with:

```python
ar.read_csv(url, nrows=1)
next(ar.read_csv_chunked(url, nrows=1, chunksize=1))
```

## Actual result

Every operation consumes the complete response before parsing:

```text
scan_csv(sample_size=1):                    2,097,152 bytes, 65 read calls
read_csv(nrows=1):                          2,097,152 bytes, 65 read calls
read_csv_chunked(nrows=1, chunksize=1):     2,097,152 bytes, 65 read calls
```

`_fetch_url_to_tempfile()` always reads until EOF, and only afterward does the
native reader apply `sample_size`, `nrows`, or `chunksize`.

## Expected result

- `scan_csv(..., sample_size=N)` should stop fetching once enough complete
  records are available for schema inference.
- Row-limited reads should avoid downloading data that cannot contribute to
  the requested result where practical.
- Remote chunked reading should either stream incrementally or clearly
  document that the entire response is first materialized.
- Remote materialization should enforce a configurable response-size limit or
  another safeguard against unbounded temporary-disk growth.

## Impact / severity

**High impact for large or untrusted remote inputs.** A schema-only request or
one-row read can transfer and write an arbitrarily large response, causing
excessive latency, bandwidth use, and temporary-disk exhaustion. A response
without `Content-Length` can continue until the filesystem fills.

## Likely affected area

- `arnio/io.py::_fetch_url_to_tempfile`
- `_materialize_csv_input`
- `scan_csv`
- `read_csv`
- `read_csv_chunked`
- `docs/remote.md`

## Suggested next action

Introduce a remote input abstraction that can stop after complete
sampled/limited records, and add a configurable maximum response size as a
fallback. Add regression tests that assert byte consumption remains bounded
for `sample_size=1`, `nrows=1`, and early chunk iteration.
