import contextlib
import importlib.metadata
import io
import json
import os
import pathlib
import sys
import tempfile
import traceback


sys.path.append(r"C:\Users\anish\anaconda3\Lib\site-packages")
_dll_handle = os.add_dll_directory(r"C:\Users\anish\anaconda3\Library\bin")

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append({"name": name, "status": "PASS", "detail": detail})
    except Exception as exc:
        results.append(
            {
                "name": name,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


import arnio as ar
import arnio._arnio_cpp as native
import pandas as pd


def wheel_origin():
    expected_site = os.environ.get("ARNIO_EXPECTED_SITE", "wheel-smoke-clean")
    assert expected_site in ar.__file__
    assert expected_site in native.__file__
    return {
        "arnio": ar.__file__,
        "native": native.__file__,
        "version": ar.__version__,
    }


check("wheel origin", wheel_origin)
check(
    "public exports",
    lambda: all(
        hasattr(ar, name)
        for name in [
            "read_csv",
            "ArFrame",
            "cast_types",
            "Schema",
            "profile",
            "pipeline",
        ]
    ),
)

with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    csv = root / "sample.csv"
    csv.write_text(
        "name,age,score\nAlice,30,9.5\nBob,bad,8.0\n",
        encoding="utf-8",
    )
    frame_box = {}

    def read_case():
        frame = ar.read_csv(str(csv))
        frame_box["frame"] = frame
        assert frame.shape == (2, 3), frame.shape
        return {
            "shape": frame.shape,
            "columns": frame.columns,
            "dtypes": frame.dtypes,
        }

    check("read_csv native", read_case)

    def scan_case():
        result = ar.scan_csv(str(csv))
        assert isinstance(result, dict), type(result)
        assert set(result) == {"name", "age", "score"}, result
        return result

    check("scan_csv", scan_case)

    def chunk_case():
        chunk_csv = root / "chunked.csv"
        chunk_csv.write_text(
            "name,age,score\nAlice,30,9.5\nBob,25,8.0\n",
            encoding="utf-8",
        )
        chunks = list(ar.read_csv_chunked(str(chunk_csv), chunksize=1))
        assert [chunk.shape for chunk in chunks] == [(1, 3), (1, 3)]
        return [chunk.shape for chunk in chunks]

    check("read_csv_chunked", chunk_case)

    def roundtrip_case():
        out = root / "roundtrip.csv"
        ar.write_csv(frame_box["frame"], str(out))
        reread = ar.read_csv(str(out))
        assert reread.shape == frame_box["frame"].shape
        return out.read_text(encoding="utf-8")

    check("write/read CSV round trip", roundtrip_case)

    def pandas_case():
        native_frame = ar.from_pandas(
            pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        )
        back = ar.to_pandas(native_frame)
        assert back.to_dict(orient="list") == {
            "x": [1, 2],
            "y": ["a", "b"],
        }
        return back.dtypes.astype(str).to_dict()

    check("pandas interop", pandas_case)

    def cast_report_case():
        native_frame = ar.from_pandas(
            pd.DataFrame({"age": ["10", "bad", "30"]})
        )
        report = ar.cast_types(
            native_frame,
            {"age": "int64"},
            errors="report",
        )
        assert isinstance(report, ar.CastReport)
        assert len(report.failures) == 1
        failure = report.failures[0]
        assert (
            failure.column,
            failure.row,
            failure.value,
            failure.target_dtype,
        ) == ("age", 1, "bad", "int64")
        out = ar.to_pandas(report.frame)
        assert out["age"].iloc[0] == 10
        assert pd.isna(out["age"].iloc[1])
        return failure.__dict__

    check("cast_types report policy", cast_report_case)

    def schema_case():
        schema = ar.Schema(
            {"age": ar.Field(dtype="int64", nullable=False)}
        )
        exported = ar.schema_to_dict(schema)
        assert exported["fields"]["age"]["dtype"] == "int64", exported
        return exported

    check("schema export", schema_case)

    def quality_case():
        report = ar.profile(
            ar.from_pandas(pd.DataFrame({"x": [1, 2, 2, None]}))
        )
        payload = report.to_dict()
        assert isinstance(payload, dict)
        return {
            "keys": sorted(payload),
            "rows": payload.get("row_count"),
        }

    check("quality profile export", quality_case)

    def entrypoint_case():
        eps = [
            ep
            for ep in importlib.metadata.entry_points(
                group="console_scripts"
            )
            if ep.name == "arnio"
        ]
        assert len(eps) == 1, eps
        assert eps[0].value == "arnio.cli:main", eps[0].value
        return {"name": eps[0].name, "value": eps[0].value}

    check("console entry point metadata", entrypoint_case)

    def cli_case():
        from arnio.cli import main

        help_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(help_out):
                main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0

        scan_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(scan_out):
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
        payload = json.loads(scan_out.getvalue())
        assert set(payload["columns"]) == {"name", "age", "score"}
        return {
            "help_first_line": help_out.getvalue().splitlines()[0],
            "scan": payload,
        }

    check("CLI parser and scan", cli_case)

print(json.dumps(results, indent=2, default=str))
failed = [result for result in results if result["status"] == "FAIL"]
raise SystemExit(1 if failed else 0)
