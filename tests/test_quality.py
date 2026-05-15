"""Tests for data quality profiling and smart cleaning."""

import datetime as dt
import decimal
import io
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import arnio as ar
from arnio._core import _DType, _Frame
from arnio.frame import ArFrame
from arnio.quality import (
    QUALITY_REPORT_COLUMNS,
    CleaningSuggestion,
    DataQualityReport,
    _validate_gate_bool,
    _validate_gate_ratio_threshold,
    _validate_gate_threshold,
)


def test_data_quality_report_suggestion_kwargs_are_json_serializable():
    report = ar.DataQualityReport(
        row_count=1,
        column_count=0,
        memory_usage=0,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns={},
        suggestions=[
            (
                "custom_step",
                {
                    "created_at": dt.datetime(2024, 1, 2, 3, 4, 5),
                    "amount": decimal.Decimal("12.34"),
                    "metadata": {"count": np.int64(1)},
                    "columns": ("name", "age"),
                    "tags": {"beta", "alpha"},
                },
            )
        ],
    )

    payload = report.to_dict()
    json.dumps(payload)

    json_output = report.to_json()
    assert isinstance(json_output, str)
    assert json.loads(json_output) == payload

    kwargs = payload["suggestions"][0]["kwargs"]
    assert kwargs["created_at"] == "2024-01-02T03:04:05"
    assert kwargs["amount"] == "12.34"
    assert kwargs["metadata"] == {"count": 1}
    assert kwargs["columns"] == ["name", "age"]
    assert kwargs["tags"] == ["alpha", "beta"]


def test_data_quality_report_to_json_round_trips_through_to_dict_options():
    report = ar.profile(
        ar.from_pandas(
            pd.DataFrame(
                {
                    "name": ["Alice", "Bob"],
                    "age": [30, 40],
                    "secret": ["token-a", "token-b"],
                }
            )
        ),
        sample_size=2,
    )

    expected = report.to_dict(
        redact_sample_values=True,
        exclude_columns=["secret"],
    )

    actual = json.loads(
        report.to_json(
            redact_sample_values=True,
            exclude_columns=["secret"],
        )
    )

    assert actual == expected


def test_data_quality_report_suggestion_kwargs_stringifies_unsupported_leaf():

    report = ar.DataQualityReport(
        row_count=1,
        column_count=0,
        memory_usage=0,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns={},
        suggestions=[
            (
                "custom_step",
                {
                    "path": Path("data/input.csv"),
                    "nested": {"complex": complex(1, 2)},
                },
            )
        ],
    )

    payload = report.to_dict()
    json.dumps(payload)

    kwargs = payload["suggestions"][0]["kwargs"]
    assert kwargs["path"] == str(Path("data/input.csv"))
    assert kwargs["nested"]["complex"] == "(1+2j)"
    assert json.loads(report.to_json()) == payload


def test_data_quality_report_suggestion_kwargs_normalizes_frozenset():
    report = ar.DataQualityReport(
        row_count=1,
        column_count=0,
        memory_usage=0,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns={},
        suggestions=[
            (
                "custom_step",
                {
                    "values": frozenset(["beta", "alpha"]),
                    "nested": {
                        "ids": frozenset([np.int64(2), np.int64(1)]),
                    },
                },
            )
        ],
    )

    payload = report.to_dict()
    json.dumps(payload)

    kwargs = payload["suggestions"][0]["kwargs"]

    assert kwargs["values"] == ["alpha", "beta"]
    assert kwargs["nested"]["ids"] == [1, 2]


def test_data_quality_report_suggestion_kwargs_normalizes_frozenset_nested_values():
    report = ar.DataQualityReport(
        row_count=1,
        column_count=0,
        memory_usage=0,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns={},
        suggestions=[
            (
                "custom_step",
                {
                    "values": frozenset(["beta", "alpha"]),
                    "nested": {
                        "values": frozenset([np.int64(2), np.int64(1)]),
                    },
                },
            )
        ],
    )

    payload = report.to_dict()
    json.dumps(payload)

    kwargs = payload["suggestions"][0]["kwargs"]
    assert kwargs["values"] == ["alpha", "beta"]
    assert kwargs["nested"]["values"] == [1, 2]
    assert json.loads(report.to_json()) == payload


def test_data_quality_report_suggestion_kwargs_recurses_numpy_item_result():
    report = ar.DataQualityReport(
        row_count=1,
        column_count=0,
        memory_usage=0,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns={},
        suggestions=[
            (
                "custom_step",
                {
                    "timestamp": np.datetime64("2024-01-02T03:04:05"),
                    "non_finite": np.float64(float("inf")),
                },
            )
        ],
    )

    payload = report.to_dict()
    json.dumps(payload)

    kwargs = payload["suggestions"][0]["kwargs"]
    assert kwargs["timestamp"].startswith("2024-01-02")
    assert kwargs["non_finite"] is None
    assert json.loads(report.to_json()) == payload


def test_data_quality_report_to_dict_normalizes_suggestions_after_exclusions():
    columns = ar.profile(
        ar.from_pandas(
            pd.DataFrame(
                {
                    "name": [" Alice ", "Bob"],
                    "secret": ["token-a", "token-b"],
                }
            )
        )
    ).columns

    report = ar.DataQualityReport(
        row_count=2,
        column_count=2,
        memory_usage=100,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns=columns,
        suggestions=[
            (
                "custom_step",
                {
                    "subset": np.array(["name", "secret"]),
                    "columns": ("name", "secret"),
                    "cast_types": {
                        "name": np.str_("string"),
                        "secret": np.str_("string"),
                    },
                    "metadata": {"kept_count": np.int64(1)},
                },
            )
        ],
    )

    result = report.to_dict(
        redact_sample_values=True,
        exclude_columns=["secret"],
    )
    json.dumps(result)

    kwargs = result["suggestions"][0]["kwargs"]

    assert kwargs["subset"] == ["name"]
    assert kwargs["columns"] == ["name"]
    assert kwargs["cast_types"] == {"name": "string"}
    assert kwargs["metadata"] == {"kept_count": 1}
    assert "secret" not in json.dumps(result)
    assert result["columns"]["name"]["sample_values"] == ["[REDACTED]", "[REDACTED]"]


def test_profile_reports_quality_signals(tmp_path):
    path = tmp_path / "quality.csv"
    path.write_text(
        "id,name,email,score\n"
        "1, Alice ,alice@test.com,95.5\n"
        "2,Bob,bob@test.com,\n"
        "2,Bob,bob@test.com,\n"
    )

    report = ar.profile(ar.read_csv(path))

    assert report.row_count == 3
    assert report.column_count == 4
    assert report.duplicate_rows == 1
    assert report.columns["name"].whitespace_count == 1
    assert report.columns["email"].semantic_type == "email"
    assert report.columns["score"].null_count == 2
    assert ("drop_duplicates", {"keep": "first"}) in report.suggestions


def test_report_summary_and_pandas_output(csv_with_whitespace):
    report = ar.profile(ar.read_csv(csv_with_whitespace))
    summary = report.summary()
    df = report.to_pandas()

    assert summary["rows"] == 3
    assert summary["columns_with_whitespace"] == ["name", "city"]
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == QUALITY_REPORT_COLUMNS
    assert set(df["name"]) == {"name", "city"}


def test_report_to_pandas_uses_stable_columns_for_empty_report():
    report = ar.DataQualityReport(
        row_count=0,
        column_count=0,
        memory_usage=0,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns={},
        suggestions=[],
    )

    df = report.to_pandas()

    assert df.columns.tolist() == QUALITY_REPORT_COLUMNS
    assert df.shape == (0, len(QUALITY_REPORT_COLUMNS))


def test_report_to_pandas_uses_stable_columns_when_all_columns_are_excluded():
    frame = ar.from_pandas(pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}))

    report = ar.profile(frame, exclude_columns=["id", "name"])
    df = report.to_pandas()

    assert report.column_count == 0
    assert report.columns == {}
    assert df.columns.tolist() == QUALITY_REPORT_COLUMNS
    assert df.shape == (0, len(QUALITY_REPORT_COLUMNS))


def test_report_to_pandas_preserves_columns_and_values_for_normal_reports():
    report = ar.profile(
        ar.from_pandas(pd.DataFrame({"name": ["Alice", "Bob"], "score": [1.0, 2.0]}))
    )

    df = report.to_pandas()

    assert df.columns.tolist() == QUALITY_REPORT_COLUMNS
    assert df["name"].tolist() == ["name", "score"]

    name_row = df.loc[df["name"] == "name"].iloc[0]
    score_row = df.loc[df["name"] == "score"].iloc[0]

    assert name_row["name"] == "name"
    assert name_row["null_count"] == 0
    assert pd.isna(name_row["q25"])
    assert score_row["name"] == "score"
    assert score_row["null_count"] == 0
    assert score_row["min"] == 1.0
    assert score_row["max"] == 2.0
    assert pd.notna(score_row["q25"])


def test_profile_numeric_quantiles():
    frame = ar.from_pandas(pd.DataFrame({"age": [1.0, 2.0, 3.0, 4.0, 5.0]}))

    report = ar.profile(frame)
    profile = report.columns["age"].to_dict()

    assert profile["q25"] == 2.0
    assert profile["q50"] == 3.0
    assert profile["q75"] == 4.0
    assert profile["q95"] == 4.8
    assert profile["iqr"] == 2.0
    assert profile["outlier_lower_bound"] == -1.0
    assert profile["outlier_upper_bound"] == 7.0
    assert profile["outlier_count"] == 0
    assert profile["outlier_ratio"] == 0.0


def test_profile_numeric_quantiles_and_outliers():
    frame = ar.from_pandas(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]}))

    report = ar.profile(frame)
    profile = report.columns["x"].to_dict()

    assert profile["q25"] == 2.0
    assert profile["q50"] == 3.0
    assert profile["q75"] == 4.0
    assert profile["q95"] == 80.8
    assert profile["iqr"] == 2.0
    assert profile["outlier_lower_bound"] == -1.0
    assert profile["outlier_upper_bound"] == 7.0
    assert profile["outlier_count"] == 1
    assert profile["outlier_ratio"] == 0.2
    assert "potential_outliers" in profile["warnings"]


def test_profile_all_null_numeric_quantiles():
    frame = ar.from_pandas(
        pd.DataFrame({"score": pd.Series([None, None], dtype="float64")})
    )

    report = ar.profile(frame)
    profile = report.columns["score"].to_dict()

    assert profile["q25"] is None
    assert profile["q50"] is None
    assert profile["q75"] is None
    assert profile["q95"] is None
    assert profile["outlier_count"] is None
    assert profile["outlier_ratio"] is None
    assert profile["iqr"] is None
    assert profile["outlier_lower_bound"] is None
    assert profile["outlier_upper_bound"] is None


def test_profile_numeric_outliers_skipped_when_too_few_values():
    frame = ar.from_pandas(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    report = ar.profile(frame)
    profile = report.columns["x"].to_dict()

    # Quartiles still computed; IQR summary needs >= 4 non-null values
    assert profile["q25"] is not None
    assert profile["outlier_count"] is None
    assert profile["outlier_ratio"] is None
    assert profile["iqr"] is None
    assert profile["outlier_lower_bound"] is None
    assert profile["outlier_upper_bound"] is None
    assert "potential_outliers" not in profile["warnings"]


def test_profile_numeric_outliers_skipped_with_two_values():
    frame = ar.from_pandas(pd.DataFrame({"x": [1.0, 100.0]}))

    report = ar.profile(frame)
    profile = report.columns["x"].to_dict()

    assert profile["outlier_count"] is None
    assert profile["outlier_ratio"] is None
    assert profile["iqr"] is None
    assert profile["outlier_lower_bound"] is None
    assert profile["outlier_upper_bound"] is None


def test_profile_numeric_zero_iqr_constant_column_no_outliers():
    frame = ar.from_pandas(pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]}))

    report = ar.profile(frame)
    profile = report.columns["x"].to_dict()

    assert profile["q25"] == profile["q75"] == 5.0
    assert profile["iqr"] == 0.0
    assert profile["outlier_lower_bound"] == 5.0
    assert profile["outlier_upper_bound"] == 5.0
    assert profile["outlier_count"] == 0
    assert profile["outlier_ratio"] == 0.0
    assert "potential_outliers" not in profile["warnings"]


def test_profile_numeric_zero_iqr_with_extreme_value():
    frame = ar.from_pandas(pd.DataFrame({"x": [10.0, 10.0, 10.0, 10.0, 50.0]}))

    report = ar.profile(frame)
    profile = report.columns["x"].to_dict()

    assert profile["q25"] == 10.0
    assert profile["q75"] == 10.0
    assert profile["iqr"] == 0.0
    assert profile["outlier_lower_bound"] == 10.0
    assert profile["outlier_upper_bound"] == 10.0
    assert profile["outlier_count"] == 1
    assert profile["outlier_ratio"] == 0.2
    assert profile["warnings"] == ["potential_outliers"]


def test_profile_non_numeric_no_quantiles():
    frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", "Bob", "Cara"]}))

    report = ar.profile(frame)
    profile = report.columns["name"].to_dict()

    assert "q25" not in profile
    assert "q50" not in profile
    assert "q75" not in profile
    assert "q95" not in profile
    assert "iqr" not in profile
    assert "outlier_lower_bound" not in profile
    assert "outlier_upper_bound" not in profile
    assert "outlier_count" not in profile
    assert "outlier_ratio" not in profile


def test_profile_email_and_url_validity_ratios():
    df = pd.DataFrame(
        {
            "good_email": [
                "alice@test.com",
                "bob@test.com",
                "cara@test.com",
                "dave@test.com",
                "eve@test.com",
            ],
            "mixed_email": [
                "alice@test.com",
                "bob@test.com",
                "cara@test.com",
                "dave@test.com",
                "invalid-email",
            ],
            "good_url": [
                "http://test.com",
                "https://example.com/foo",
                "https://another.org",
                "http://a.b",
                "https://last.com",
            ],
            "mixed_url": [
                "http://test.com",
                "https://example.com/foo",
                "https://another.org",
                "http://a.b",
                "not-a-url",
            ],
            "generic": ["hello", "world", "foo", "bar", "baz"],
        }
    )

    frame = ar.from_pandas(df)
    report = ar.profile(frame)

    assert report.columns["good_email"].semantic_type == "email"
    assert report.columns["mixed_email"].semantic_type == "email"
    assert report.columns["good_url"].semantic_type == "url"
    assert report.columns["mixed_url"].semantic_type == "url"
    assert report.columns["generic"].semantic_type == "categorical"

    assert report.columns["good_email"].email_validity_ratio == 1.0
    assert report.columns["good_email"].url_validity_ratio is None

    assert report.columns["mixed_email"].email_validity_ratio == 0.8
    assert report.columns["mixed_email"].url_validity_ratio is None

    assert report.columns["good_url"].url_validity_ratio == 1.0
    assert report.columns["good_url"].email_validity_ratio is None

    assert report.columns["mixed_url"].url_validity_ratio == 0.8
    assert report.columns["mixed_url"].email_validity_ratio is None

    assert report.columns["generic"].email_validity_ratio is None
    assert report.columns["generic"].url_validity_ratio is None

    good_email_dict = report.columns["good_email"].to_dict()
    assert good_email_dict["email_validity_ratio"] == 1.0
    assert good_email_dict["url_validity_ratio"] is None

    mixed_url_dict = report.columns["mixed_url"].to_dict()
    assert mixed_url_dict["url_validity_ratio"] == 0.8
    assert mixed_url_dict["email_validity_ratio"] is None

    pdf = report.to_pandas()
    good_email_row = pdf[pdf["name"] == "good_email"].iloc[0]
    assert good_email_row["email_validity_ratio"] == 1.0
    assert (
        pd.isna(good_email_row["url_validity_ratio"])
        or good_email_row["url_validity_ratio"] is None
    )


def test_compare_profiles_identical_profiles_are_ok():
    frame = ar.from_pandas(
        pd.DataFrame({"score": [10.0, 11.0, 12.0], "city": ["a", "b", "a"]})
    )

    comparison = ar.compare_profiles(ar.profile(frame), ar.profile(frame))

    assert set(comparison.drift_report) == {"score", "city"}
    assert all(entry["status"] == "ok" for entry in comparison.drift_report.values())
    assert comparison.status_counts == {"ok": 2, "warning": 0, "changed": 0}


def test_compare_profiles_detects_numeric_drift():
    baseline = ar.profile(ar.from_pandas(pd.DataFrame({"score": [10.0, 10.0, 10.0]})))
    current = ar.profile(ar.from_pandas(pd.DataFrame({"score": [20.0, 20.0, 20.0]})))

    comparison = ar.compare_profiles(baseline, current)

    assert comparison.drift_report["score"]["status"] in {"warning", "changed"}
    assert comparison.drift_report["score"]["changes"]["mean"]["baseline"] == 10.0
    assert comparison.drift_report["score"]["changes"]["mean"]["comparison"] == 20.0


def test_compare_profiles_rejects_schema_mismatch():
    left = ar.profile(ar.from_pandas(pd.DataFrame({"score": [1.0, 2.0]})))
    right = ar.profile(
        ar.from_pandas(pd.DataFrame({"score": [1.0, 2.0], "city": ["a", "b"]}))
    )

    with pytest.raises(ValueError, match="incompatible schemas"):
        ar.compare_profiles(left, right)


def test_compare_profiles_handles_empty_profiles():
    empty = ar.profile(ar.from_pandas(pd.DataFrame()))

    comparison = ar.compare_profiles(empty, empty)

    assert comparison.drift_report == {}
    assert comparison.status_counts == {"ok": 0, "warning": 0, "changed": 0}


def test_compare_profiles_handles_single_column_profiles():
    frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", "Bob"]}))

    comparison = ar.compare_profiles(ar.profile(frame), ar.profile(frame))

    assert comparison.drift_report["name"]["status"] == "ok"
    assert comparison.status_counts == {"ok": 1, "warning": 0, "changed": 0}


def test_profile_comparison_accepts_valid_status_counts():
    base = ar.DataQualityReport(0, 0, 0, 0, 0.0, {})

    comparison = ar.ProfileComparison(
        base,
        base,
        {},
        {"ok": 1, "warning": 2, "changed": 0},
    )

    assert comparison.status_counts == {
        "ok": 1,
        "warning": 2,
        "changed": 0,
    }


def test_profile_comparison_rejects_bool_status_counts():
    base = ar.DataQualityReport(0, 0, 0, 0, 0.0, {})

    with pytest.raises(
        TypeError,
        match="status_counts values must not be booleans",
    ):
        ar.ProfileComparison(
            base,
            base,
            {},
            {"passed": True},
        )


def test_profile_comparison_rejects_negative_status_counts():
    base = ar.DataQualityReport(0, 0, 0, 0, 0.0, {})

    with pytest.raises(
        ValueError,
        match="status_counts values must be non-negative integers",
    ):
        ar.ProfileComparison(
            base,
            base,
            {},
            {"failed": -1},
        )


def test_check_quality_gates_passes_identical_profiles():
    frame = ar.from_pandas(
        pd.DataFrame({"score": [10.0, 11.0, 12.0], "city": ["a", "b", "a"]})
    )

    result = ar.check_quality_gates(ar.profile(frame), ar.profile(frame))

    assert result.passed is True
    assert result.issues == []
    assert result.summary()["passed"] is True
    assert result.to_dict()["passed"] is True
    assert result.to_dict()["summary"]["issue_count"] == 0
    assert "All configured quality gates passed" in result.to_markdown()


def test_check_quality_gates_detects_row_duplicate_null_and_numeric_drift():
    baseline = ar.profile(
        ar.from_pandas(
            pd.DataFrame({"score": [10.0, 10.0, 10.0], "city": ["a", "b", "c"]})
        )
    )
    current = ar.profile(
        ar.from_pandas(
            pd.DataFrame(
                {
                    "score": [20.0, 20.0, None, None, 20.0],
                    "city": ["a", "a", "a", "a", "a"],
                }
            )
        )
    )

    result = ar.check_quality_gates(
        baseline,
        current,
        max_row_count_delta_ratio=0.2,
        max_duplicate_ratio_delta=0.1,
        max_null_ratio_delta=0.1,
        max_numeric_mean_delta_ratio=0.5,
    )

    metrics = {issue.metric for issue in result.issues}
    assert result.passed is False
    assert {"row_count", "duplicate_ratio", "null_ratio", "numeric_mean"} <= metrics
    assert any(issue.column == "score" for issue in result.issues)


def test_check_quality_gates_detects_schema_and_dtype_changes():
    baseline = ar.profile(
        ar.from_pandas(pd.DataFrame({"score": [1, 2], "city": ["a", "b"]}))
    )
    current = ar.profile(
        ar.from_pandas(pd.DataFrame({"score": ["1", "2"], "state": ["a", "b"]}))
    )

    result = ar.check_quality_gates(baseline, current)

    issues = {(issue.metric, issue.column) for issue in result.issues}
    assert ("missing_column", "city") in issues
    assert ("new_column", "state") in issues
    assert ("dtype", "score") in issues


def test_check_quality_gates_can_allow_schema_changes_and_disable_thresholds():
    baseline = ar.profile(ar.from_pandas(pd.DataFrame({"score": [1.0, 2.0]})))
    current = ar.profile(
        ar.from_pandas(pd.DataFrame({"score": [100.0, 200.0], "extra": ["x", "y"]}))
    )

    result = ar.check_quality_gates(
        baseline,
        current,
        allow_new_columns=True,
        max_numeric_mean_delta_ratio=None,
        max_numeric_std_delta_ratio=None,
    )

    assert result.passed is True


def test_check_quality_gates_markdown_escapes_table_cells():
    baseline = ar.profile(ar.from_pandas(pd.DataFrame({"bad|name": [1, 2]})))
    current = ar.profile(ar.from_pandas(pd.DataFrame({"other": [1, 2]})))

    markdown = ar.check_quality_gates(baseline, current).to_markdown()

    assert "bad\\|name" in markdown


def test_check_quality_gates_validates_thresholds_and_flags():
    report = ar.profile(ar.from_pandas(pd.DataFrame({"score": [1.0, 2.0]})))

    with pytest.raises(ValueError, match="finite non-negative"):
        ar.check_quality_gates(report, report, max_null_ratio_delta=-0.1)

    with pytest.raises(ValueError, match="must be a ratio between 0.0 and 1.0"):
        ar.check_quality_gates(report, report, max_null_ratio_delta=1.5)

    with pytest.raises(ValueError, match="must be a ratio between 0.0 and 1.0"):
        ar.check_quality_gates(report, report, max_duplicate_ratio_delta=1.0001)

    result = ar.check_quality_gates(report, report, max_row_count_delta_ratio=2.5)
    assert result.passed is True

    with pytest.raises(TypeError, match="allow_new_columns must be a bool"):
        ar.check_quality_gates(report, report, allow_new_columns="yes")


def test_quality_gate_result_raise_for_failures():
    baseline = ar.profile(ar.from_pandas(pd.DataFrame({"score": [1.0, 2.0]})))
    current = ar.profile(ar.from_pandas(pd.DataFrame({"score": [100.0, 200.0]})))

    result = ar.check_quality_gates(
        baseline,
        current,
        max_numeric_mean_delta_ratio=0.1,
    )

    with pytest.raises(ValueError, match="data quality gate"):
        result.raise_for_failures()


def test_suggest_cleaning_returns_pipeline_compatible_steps(csv_with_duplicates):
    frame = ar.read_csv(csv_with_duplicates)
    suggestions = ar.suggest_cleaning(frame)

    assert suggestions == [("drop_duplicates", {"keep": "first"})]
    clean = ar.pipeline(frame, suggestions)
    assert clean.shape == (3, 2)


def test_suggest_cleaning_confidence_metadata():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 3],
            "name": ["Alice ", "Bob", "Charlie ", "Charlie "],
            "active": ["true", "false", "true", "true"],
            "duplicates": [1, 1, 1, 1],
        }
    )
    frame = ar.from_pandas(df)
    report = ar.profile(frame)
    suggestions = ar.suggest_cleaning(report)

    # Convert to standard list of step names to find the specific suggestions
    step_names = [s[0] for s in suggestions]

    # Check strip_whitespace
    assert "strip_whitespace" in step_names
    strip_sug = next(s for s in suggestions if s[0] == "strip_whitespace")
    assert getattr(strip_sug, "confidence_score") == 0.95
    assert "Trimming leading/trailing whitespace" in getattr(
        strip_sug, "confidence_reason"
    )
    assert getattr(strip_sug, "step") == "strip_whitespace"
    assert getattr(strip_sug, "kwargs") == {"subset": ["name"]}

    # Check cast_types
    assert "cast_types" in step_names
    cast_sug = next(s for s in suggestions if s[0] == "cast_types")
    assert getattr(cast_sug, "confidence_score") == 0.95
    assert "conforms perfectly to bool structure" in getattr(
        cast_sug, "confidence_reason"
    )

    # Check drop_duplicates
    assert "drop_duplicates" in step_names
    drop_sug = next(s for s in suggestions if s[0] == "drop_duplicates")
    # Duplicate ratio here is 1 duplicate out of 4 rows = 0.25 <= 0.5
    assert getattr(drop_sug, "confidence_score") == 0.95
    assert "Low duplicate ratio" in getattr(drop_sug, "confidence_reason")

    # Check JSON serialization of confidence metadata
    report_dict = report.to_dict()
    dict_suggestions = report_dict["suggestions"]
    assert len(dict_suggestions) == 3
    for s in dict_suggestions:
        assert "confidence_score" in s
        assert "confidence_reason" in s
        assert isinstance(s["confidence_score"], float)
        assert isinstance(s["confidence_reason"], str)

    # Check Markdown formatting
    md = report.to_markdown()
    assert "(Confidence: 0.95 -" in md


def test_cleaning_suggestion_backward_compatibility():
    """Prove backward compatibility with the existing tuple contract."""
    from arnio.quality import CleaningSuggestion

    sug = CleaningSuggestion("drop_duplicates", {"keep": "first"}, 0.9, "reason")

    # It should equate to the exact 2-tuple.
    assert sug == ("drop_duplicates", {"keep": "first"})

    # It should unpack correctly into 2 variables.
    step, kwargs = sug
    assert step == "drop_duplicates"
    assert kwargs == {"keep": "first"}

    # It should work natively with ar.pipeline
    df = pd.DataFrame(
        {
            "id": [1, 2, 2],
        }
    )
    frame = ar.from_pandas(df)
    clean = ar.pipeline(frame, [sug])
    assert clean.shape == (2, 1)


def test_auto_clean_safe_trims_without_dropping_duplicates(tmp_path):
    path = tmp_path / "safe.csv"
    path.write_text("name\n Alice \n Alice \n")

    frame = ar.read_csv(path)
    clean, report = ar.auto_clean(frame, return_report=True)
    df = ar.to_pandas(clean)

    assert report.duplicate_rows == 1
    assert clean.shape == (2, 1)
    assert list(df["name"]) == ["Alice", "Alice"]


def test_auto_clean_strict_applies_exact_deduplication(tmp_path):
    path = tmp_path / "strict.csv"
    path.write_text("name\n Alice \n Alice \n")

    clean = ar.auto_clean(ar.read_csv(path), mode="strict")

    assert clean.shape == (1, 1)


def test_auto_clean_strict_drops_duplicates_created_by_whitespace():
    frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", " Alice "]}))

    clean = ar.auto_clean(frame, mode="strict")
    result = ar.to_pandas(clean)

    assert clean.shape == (1, 1)
    assert list(result["name"]) == ["Alice"]


def test_auto_clean_strict_drops_duplicates_created_by_casts():
    frame = ar.from_pandas(pd.DataFrame({"amount": ["1", "1.0"]}))
    report = ar.auto_clean(frame, mode="strict", dry_run=True)

    clean = ar.auto_clean(
        frame,
        mode="strict",
        allow_lossy_casts=True,
        confirmed_casts=dict(report.suggestions)["cast_types"],
    )
    result = ar.to_pandas(clean)

    assert clean.shape == (1, 1)
    assert list(result["amount"]) == [1.0]
    assert pd.api.types.is_float_dtype(result["amount"])


def test_auto_clean_strict_casts_require_explicit_opt_in():
    frame = ar.from_pandas(pd.DataFrame({"active": ["true", "false"]}))

    with pytest.raises(ValueError, match="would apply type casts"):
        ar.auto_clean(frame, mode="strict")


def test_auto_clean_strict_casts_require_preview_confirmation():
    frame = ar.from_pandas(pd.DataFrame({"active": ["true", "false"]}))

    with pytest.raises(ValueError, match="requires confirmed_casts"):
        ar.auto_clean(frame, mode="strict", allow_lossy_casts=True)


def test_auto_clean_strict_rejects_mismatched_cast_confirmation():
    frame = ar.from_pandas(pd.DataFrame({"active": ["true", "false"]}))

    with pytest.raises(ValueError, match="must match the proposed cast mapping"):
        ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={"active": "int64"},
        )


@pytest.mark.parametrize(
    "confirmed_casts",
    [
        [("active", "bool")],
        {1: "bool"},
        {"active": 1},
    ],
    ids=["list", "non-string-key", "non-string-value"],
)
def test_auto_clean_rejects_invalid_confirmed_casts(confirmed_casts):
    frame = ar.from_pandas(pd.DataFrame({"active": ["true", "false"]}))

    with pytest.raises(TypeError, match="confirmed_casts"):
        ar.auto_clean(frame, mode="strict", confirmed_casts=confirmed_casts)


def test_exclude_columns_prevents_leakage_in_json():
    import json

    frame = ar.from_pandas(pd.DataFrame({"secret_token": ["true", "false"]}))
    report = ar.profile(frame)

    report_dict = report.to_dict(exclude_columns=["secret_token"])
    json_str = json.dumps(report_dict)

    assert "secret_token" not in json_str


def test_auto_clean_dry_run_returns_report_without_mutating():
    frame = ar.from_pandas(pd.DataFrame({"active": ["true", "false"]}))

    report = ar.auto_clean(frame, mode="strict", dry_run=True)

    assert isinstance(report, ar.DataQualityReport)
    assert ("cast_types", {"active": "bool"}) in report.suggestions
    assert frame.dtypes["active"] == "string"


def test_auto_clean_strict_casts_after_explicit_preview_confirmation():
    frame = ar.from_pandas(pd.DataFrame({"active": ["true", "false"]}))
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    confirmed_casts = dict(report.suggestions)["cast_types"]

    clean = ar.auto_clean(
        frame,
        mode="strict",
        allow_lossy_casts=True,
        confirmed_casts=confirmed_casts,
    )
    result = ar.to_pandas(clean)

    assert list(result["active"]) == [True, False]
    assert pd.api.types.is_bool_dtype(result["active"])


def test_auto_clean_dry_run_with_return_report_raises():
    frame = ar.from_pandas(pd.DataFrame({"name": [" Alice ", " Bob "]}))

    with pytest.raises(
        ValueError, match="return_report=True cannot be used with dry_run=True"
    ):
        ar.auto_clean(frame, dry_run=True, return_report=True)


@pytest.mark.parametrize(
    "value",
    [
        "yes",
        1,
        None,
        [],
        object(),
    ],
    ids=["string", "integer", "none", "list", "object"],
)
def test_auto_clean_rejects_non_boolean_return_report(value):
    frame = ar.from_pandas(pd.DataFrame({"name": [" Alice ", " Bob "]}))

    with pytest.raises(TypeError, match="return_report must be a bool"):
        ar.auto_clean(frame, return_report=value)


def test_auto_clean_accepts_boolean_return_report_values():
    frame = ar.from_pandas(pd.DataFrame({"name": [" Alice ", " Bob "]}))

    clean_only = ar.auto_clean(frame, return_report=False)
    clean_with_report = ar.auto_clean(frame, return_report=True)

    assert isinstance(clean_only, ar.ArFrame)
    assert isinstance(clean_with_report, tuple)
    assert len(clean_with_report) == 2
    cleaned, report = clean_with_report
    assert isinstance(cleaned, ar.ArFrame)
    assert isinstance(report, ar.DataQualityReport)


def test_auto_clean_rejects_unknown_mode(sample_csv):
    frame = ar.read_csv(sample_csv)

    try:
        ar.auto_clean(frame, mode="wild")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "mode must be" in str(exc)


def test_profile_sample_size(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("id\n1\n2\n3\n4\n5\n6\n7\n")
    frame = ar.read_csv(path)

    report_default = ar.profile(frame)
    assert len(report_default.columns["id"].sample_values) == 5

    report_custom = ar.profile(frame, sample_size=3)
    assert len(report_custom.columns["id"].sample_values) == 3

    report_zero = ar.profile(frame, sample_size=0)
    assert len(report_zero.columns["id"].sample_values) == 0


def test_profile_sample_size_small_dataset_and_nulls(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("id\n1\n\n3\n")
    frame = ar.read_csv(path)

    report = ar.profile(frame, sample_size=5)
    assert len(report.columns["id"].sample_values) == 2
    assert report.columns["id"].sample_values == [1.0, 3.0]


def test_profile_sample_size_validation(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("id\n1\n")
    frame = ar.read_csv(path)

    try:
        ar.profile(frame, sample_size=-1)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "sample_size must be non-negative" in str(exc)

    try:
        ar.profile(frame, sample_size="5")
        assert False, "Expected TypeError"
    except TypeError as exc:
        assert "sample_size must be an integer" in str(exc)
