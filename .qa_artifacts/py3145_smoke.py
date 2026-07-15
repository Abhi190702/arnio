import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile

import pandas as pd

import arnio as ar
import arnio._arnio_cpp as native


expected = os.environ["ARNIO_PY314_SITE"]
assert expected in ar.__file__, ar.__file__
assert expected in native.__file__, native.__file__
assert sys.version_info[:2] == (3, 14), sys.version

with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    csv = root / "input.csv"
    csv.write_text(
        "name,age,score\nAlice,30,9.5\nBob,bad,8.0\n",
        encoding="utf-8",
    )

    frame = ar.read_csv(csv)
    assert frame.shape == (2, 3)
    assert frame.dtypes == {
        "name": "string",
        "age": "string",
        "score": "float64",
    }

    out = root / "out.csv"
    ar.write_csv(frame, out)
    assert ar.read_csv(out).shape == frame.shape

    pandas_frame = ar.from_pandas(
        pd.DataFrame({"age": ["10", "bad", "30"]})
    )
    report = ar.cast_types(
        pandas_frame,
        {"age": "int64"},
        errors="report",
    )
    assert len(report.failures) == 1
    assert report.failures[0].row == 1
    assert pd.isna(ar.to_pandas(report.frame)["age"].iloc[1])

    schema = ar.Schema(
        {"age": ar.Field(dtype="int64", nullable=False)}
    )
    assert ar.schema_to_dict(schema)["fields"]["age"]["dtype"] == "int64"

    profile = ar.profile(
        ar.from_pandas(pd.DataFrame({"x": [1, 2, 2, None]}))
    )
    assert profile.to_dict()["row_count"] == 4

    from arnio.cli import main

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            main(
                [
                    "scan",
                    "--input",
                    str(csv),
                    "--format",
                    "json",
                ]
            )
    except SystemExit as exc:
        assert exc.code == 0

    assert set(json.loads(stdout.getvalue())["columns"]) == {
        "name",
        "age",
        "score",
    }

print(
    json.dumps(
        {
            "python": sys.version,
            "arnio": ar.__file__,
            "native": native.__file__,
            "version": ar.__version__,
            "status": "PASS",
        },
        indent=2,
    )
)
