from __future__ import annotations

import sys
from unittest.mock import patch

sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ != "_arnio_editable"
]

import arnio as ar


CHUNK_SIZE = 32 * 1024
CHUNK_COUNT = 64


class CountingResponse:
    def __init__(self) -> None:
        first_rows = (CHUNK_SIZE - len(b"value\n")) // 2
        self._chunks = [b"value\n" + (b"1\n" * first_rows)]
        self._chunks.extend([b"1\n" * (CHUNK_SIZE // 2)] * (CHUNK_COUNT - 1))
        self.read_calls = 0
        self.bytes_read = 0

    def __enter__(self) -> "CountingResponse":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self, _size: int = -1) -> bytes:
        self.read_calls += 1
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        self.bytes_read += len(chunk)
        return chunk


def run_probe(name: str, operation) -> None:
    response = CountingResponse()
    with patch("arnio.io.urllib.request.urlopen", return_value=response):
        result = operation()
    print(
        f"{name}: bytes_read={response.bytes_read}, "
        f"read_calls={response.read_calls}, result={result}"
    )


print(f"arnio={ar.__file__}")
run_probe(
    "scan_csv(sample_size=1)",
    lambda: ar.scan_csv("https://example.invalid/large.csv", sample_size=1),
)
run_probe(
    "read_csv(nrows=1)",
    lambda: ar.read_csv("https://example.invalid/large.csv", nrows=1).shape,
)
run_probe(
    "read_csv_chunked(nrows=1,chunksize=1)",
    lambda: next(
        ar.read_csv_chunked(
            "https://example.invalid/large.csv",
            nrows=1,
            chunksize=1,
        )
    ).shape,
)
