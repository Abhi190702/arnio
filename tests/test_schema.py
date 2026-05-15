"""Tests for schema validation."""

import pytest

import arnio as ar
from arnio.frame import ArFrame
from arnio.schema import (
    _SEMANTIC_PATTERNS,
    Field,
    Schema,
    _is_safely_convertible_to_dtype,
    validate,
)


def test_dtype_validation_reports_safe_int_conversion_for_numeric_strings():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "age": pd.Series(
                    ["1", "2", "3"],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema({"age": ar.Int64()})

    result = ar.validate(frame, schema)

    assert not result.passed
    assert "safely convertible to 'int64'" in result.issues[0].message


def test_dtype_validation_reports_safe_float_conversion_for_numeric_strings():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "score": pd.Series(
                    ["1.5", "2.0", "3.25"],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema({"score": ar.Float64()})

    result = ar.validate(frame, schema)

    assert not result.passed
    assert "safely convertible to 'float64'" in result.issues[0].message


def test_schema_validation_row_indexed_issues_respect_cap():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "name": [None, None, "ok"],
            }
        )
    )

    schema = ar.Schema(
        {
            "name": ar.Field(nullable=False),
        }
    )

    result = ar.validate(frame, schema, max_errors=1)

    assert result.issue_count == 1
    assert result.bad_rows == [1]


def test_dtype_validation_does_not_report_safe_conversion_for_invalid_numeric_strings():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "age": pd.Series(
                    ["1", "abc", "3"],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema({"age": ar.Int64()})

    result = ar.validate(frame, schema)

    assert not result.passed
    assert "safely convertible" not in result.issues[0].message


def test_validate_rejects_chunked_iterators(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("email\n" "a@example.com\n")

    chunks = ar.read_csv_chunked(path, chunksize=1)

    with pytest.raises(
        TypeError, match="Chunked validation is not currently supported"
    ):
        ar.validate(chunks, {"email": ar.Email(nullable=False)})


def test_dtype_validation_does_not_report_safe_conversion_for_identifier_like_columns():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "user_id": pd.Series(
                    ["001", "002", "003"],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema({"user_id": ar.Int64()})

    result = ar.validate(frame, schema)

    assert not result.passed
    assert "safely convertible" not in result.issues[0].message


def test_dtype_validation_does_not_report_safe_conversion_for_empty_strings():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "age": pd.Series(
                    [None, None],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema({"age": ar.Int64()})

    result = ar.validate(frame, schema)

    assert not result.passed
    assert "safely convertible" not in result.issues[0].message


def test_dtype_validation_preserves_warning_severity_for_numeric_strings():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "age": pd.Series(
                    ["1", "2", "3"],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema(
        {
            "age": ar.Int64(severity="warning"),
        }
    )

    result = ar.validate(frame, schema)

    assert result.issues[0].severity == "warning"


def test_dtype_validation_does_not_report_safe_conversion_above_int64_max():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "value": pd.Series(
                    ["9223372036854775808"],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema({"value": ar.Int64()})

    result = ar.validate(frame, schema)

    assert not result.passed
    assert "safely convertible" not in result.issues[0].message


def test_dtype_validation_does_not_report_safe_conversion_below_int64_min():
    frame = ar.from_pandas(
        pd.DataFrame(
            {
                "value": pd.Series(
                    ["-9223372036854775809"],
                    dtype="string",
                )
            }
        )
    )

    schema = ar.Schema({"value": ar.Int64()})

    result = ar.validate(frame, schema)

    assert not result.passed
    assert "safely convertible" not in result.issues[0].message


def test_schema_validation_passes_for_valid_frame(sample_csv):
    frame = ar.read_csv(sample_csv)
    schema = ar.Schema(
        {
            "name": ar.String(nullable=False, min_length=3),
            "age": ar.Int64(nullable=False, min=0, max=120),
            "email": ar.Email(nullable=False, unique=True),
            "active": ar.Bool(nullable=False),
        },
        strict=True,
    )

    result = ar.validate(frame, schema)

    assert result.passed
    assert result.issue_count == 0
    assert result.bad_rows == []


def test_schema_validation_stops_after_max_errors(tmp_path):
    path = tmp_path / "bad.csv"

    path.write_text(
        "name,age,email\n"
        ",150,invalid-email\n"
        ",200,another-invalid\n"
        ",300,bad-email\n"
    )

    frame = ar.read_csv(path)
    schema = ar.Schema(
        {
            "name": ar.String(nullable=False),
            "age": ar.Int64(min=0, max=120),
            "email": ar.Email(nullable=False),
        }
    )

    result = ar.validate(frame, schema, max_errors=2)

    assert result.issue_count == 2
    assert len(result.issues) == 2


def test_schema_rejects_invalid_field_values_string(sample_csv):
    frame = ar.read_csv(sample_csv)
    with pytest.raises(TypeError, match="must be a Field instance"):
        ar.validate(frame, {"id": "int64"})


def test_schema_rejects_invalid_field_values_dict(sample_csv):
    frame = ar.read_csv(sample_csv)
    with pytest.raises(TypeError, match="must be a Field instance"):
        ar.validate(frame, {"id": {"type": "int64"}})


def test_schema_rejects_invalid_field_values_none(sample_csv):
    frame = ar.read_csv(sample_csv)
    with pytest.raises(TypeError, match="must be a Field instance"):
        ar.validate(frame, {"id": None})


def test_schema_rejects_non_string_field_name_integer(sample_csv):
    frame = ar.read_csv(sample_csv)
    with pytest.raises(TypeError, match="Schema field names must be strings"):
        ar.validate(frame, {1: ar.String()})


def test_schema_rejects_non_string_field_name_none(sample_csv):
    frame = ar.read_csv(sample_csv)
    with pytest.raises(TypeError, match="Schema field names must be strings"):
        ar.validate(frame, {None: ar.String()})


def test_schema_rejects_non_string_field_name_tuple(sample_csv):
    frame = ar.read_csv(sample_csv)
    with pytest.raises(TypeError, match="Schema field names must be strings"):
        ar.validate(frame, {("a", "b"): ar.String()})


def test_schema_validation_collects_row_level_issues(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "name,age,email,status\n"
        "Alice,30,alice@test.com,active\n"
        ",150,not-an-email,blocked\n"
        "Bob,-1,bob@test.com,unknown\n"
    )
    frame = ar.read_csv(path)
    schema = ar.Schema(
        {
            "name": ar.String(nullable=False),
            "age": ar.Int64(nullable=False, min=0, max=120),
            "email": ar.Email(nullable=False),
            "status": ar.String(allowed={"active", "blocked"}),
        }
    )

    result = schema.validate(frame)
    rules = {issue.rule for issue in result.issues}

    assert not result.passed
    assert result.bad_rows == [2, 3]
    assert {"nullable", "max", "min", "email", "allowed"} <= rules
    assert result.summary()["issues_by_column"]["age"] == 2


def test_string_allowed_is_case_sensitive_by_default(tmp_path):
    path = tmp_path / "status.csv"
    path.write_text("status\nactive\nACTIVE\nActive\n")

    result = ar.validate(
        ar.read_csv(path),
        {"status": ar.String(allowed=["active"])},
    )

    assert not result.passed
    assert result.issue_count == 2
    assert [issue.row_index for issue in result.issues] == [2, 3]


def test_string_case_sensitive_round_trips_through_json():
    schema = ar.Schema({"status": ar.String(allowed=["active"], case_sensitive=False)})

    restored = ar.Schema.from_json(schema.to_json())

    assert restored.fields["status"].case_sensitive is False


def test_string_allowed_supports_case_insensitive_matching(tmp_path):
    path = tmp_path / "status.csv"
    path.write_text("status\nactive\nACTIVE\nActive\ninactive\n")

    result = ar.validate(
        ar.read_csv(path),
        {
            "status": ar.String(
                allowed=["active", "inactive"],
                case_sensitive=False,
            )
        },
    )

    assert result.passed
    assert result.issue_count == 0


def test_string_allowed_case_insensitive_rejects_invalid_values(tmp_path):
    path = tmp_path / "status.csv"
    path.write_text("status\nactive\nACTIVE\npending\n")

    result = ar.validate(
        ar.read_csv(path),
        {
            "status": ar.String(
                allowed=["active"],
                case_sensitive=False,
            )
        },
    )

    assert not result.passed
    assert result.issue_count == 1
    assert result.issues[0].row_index == 3
    assert result.issues[0].rule == "allowed"


def test_string_case_sensitive_must_be_bool():
    with pytest.raises(TypeError, match="case_sensitive must be a bool"):
        ar.String(allowed=["active"], case_sensitive="false")


def test_schema_reports_missing_and_unexpected_columns(sample_csv):
    frame = ar.read_csv(sample_csv)
    schema = ar.Schema({"missing": ar.String()}, strict=True)

    result = ar.validate(frame, schema)
    rules = [issue.rule for issue in result.issues]

    assert "required_column" in rules
    assert "unexpected_column" in rules


# --- ValidationResult constructor validation (regression for #1684) ---


def test_validation_result_rejects_string_row_count():
    with pytest.raises(TypeError, match="row_count"):
        ar.ValidationResult(row_count="1", issue_count=0, issues=[])


def test_validation_result_rejects_negative_row_count():
    with pytest.raises(ValueError, match="row_count"):
        ar.ValidationResult(row_count=-1, issue_count=0, issues=[])


def test_validation_result_rejects_bool_row_count():
    with pytest.raises(TypeError, match="row_count"):
        ar.ValidationResult(row_count=True, issue_count=0, issues=[])


def test_validation_result_rejects_string_issue_count():
    with pytest.raises(TypeError, match="issue_count"):
        ar.ValidationResult(row_count=1, issue_count="0", issues=[])


def test_validation_result_rejects_negative_issue_count():
    with pytest.raises(ValueError, match="issue_count"):
        ar.ValidationResult(row_count=1, issue_count=-1, issues=[])


def test_validation_result_rejects_bool_issue_count():
    with pytest.raises(TypeError, match="issue_count"):
        ar.ValidationResult(row_count=1, issue_count=False, issues=[])


def test_validation_result_rejects_non_list_issues():
    with pytest.raises(TypeError, match="issues"):
        ar.ValidationResult(row_count=1, issue_count=0, issues=None)


def test_validation_result_rejects_string_item_in_issues():
    with pytest.raises(TypeError, match="issues"):
        ar.ValidationResult(row_count=1, issue_count=1, issues=["bad"])


def test_validation_result_rejects_string_bad_rows():
    with pytest.raises(TypeError, match="bad_rows"):
        ar.ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows="abc")


def test_validation_result_rejects_negative_bad_rows_entry():
    with pytest.raises(ValueError, match="bad_rows"):
        ar.ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows=[-1])


def test_validation_result_rejects_non_int_bad_rows_entry():
    with pytest.raises(TypeError, match="bad_rows"):
        ar.ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows=["1"])


def test_validation_result_rejects_mismatched_issue_count():
    issue = ar.ValidationIssue(column="x", rule="dtype", message="bad type")
    with pytest.raises(ValueError, match="issue_count"):
        ar.ValidationResult(row_count=1, issue_count=2, issues=[issue])


def test_validation_result_valid_construction():
    issue = ar.ValidationIssue(column="x", rule="dtype", message="bad type")
    result = ar.ValidationResult(
        row_count=5,
        issue_count=1,
        issues=[issue],
        bad_rows=[0],
    )
    assert result.row_count == 5
    assert result.issue_count == 1
    assert len(result.issues) == 1
    assert result.bad_rows == [0]


# --- end ValidationResult constructor validation ---


def test_validation_result_to_pandas_empty_has_stable_columns():
    result = ar.ValidationResult(
        row_count=3,
        issue_count=0,
        issues=[],
        bad_rows=[],
    )

    df = result.to_pandas()

    assert df.empty
    assert list(df.columns) == [
        "column",
        "rule",
        "message",
        "row_index",
        "value",
        "severity",
    ]


def test_schema_validation_bool_max_errors_rejected():
    frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
    schema = ar.Schema({"a": ar.Field()})

    with pytest.raises(TypeError, match="max_errors"):
        ar.validate(frame, schema, max_errors=True)


def test_schema_validation_float_max_errors_rejected():
    frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
    schema = ar.Schema({"a": ar.Field()})

    with pytest.raises(TypeError, match="max_errors"):
        ar.validate(frame, schema, max_errors=1.5)


def test_schema_validation_custom_rule_respects_max_errors():
    def bad_rule(df):
        return [
            ar.ValidationIssue(
                column="a",
                rule="custom",
                message="error 1",
                row_index=1,
            ),
            ar.ValidationIssue(
                column="a",
                rule="custom",
                message="error 2",
                row_index=2,
            ),
        ]

    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))

    schema = ar.Schema(
        {"a": ar.Field()},
        rules=[bad_rule],
    )

    result = ar.validate(frame, schema, max_errors=1)

    assert result.issue_count == 1
    assert result.bad_rows == [1]


def test_validation_result_summary_counts_repeated_issues_in_one_column():
    result = ar.ValidationResult(
        row_count=3,
        issue_count=3,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=0
            ),
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=1
            ),
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=2
            ),
        ],
        bad_rows=[0, 1, 2],
    )

    summary = result.summary()

    assert summary["issues_by_rule"] == {"min": 3}
    assert summary["issues_by_column"] == {"age": 3}
    assert summary["issues_by_column_and_rule"] == {"age": {"min": 3}}


def test_schema_validation_negative_max_errors(tmp_path):
    path = tmp_path / "data.csv"

    path.write_text("name\njohn\n")

    frame = ar.read_csv(path)

    schema = ar.Schema(
        {
            "name": ar.String(),
        }
    )

    with pytest.raises(ValueError):
        ar.validate(frame, schema, max_errors=-1)


def test_schema_validation_unique_missing_columns_respects_max_errors():
    frame = ar.read_csv(io.StringIO("x\n1\n"))

    schema = ar.Schema(
        {},
        unique=["a", "b"],
    )

    result = ar.validate(frame, schema, max_errors=1)

    assert result.issue_count == 1


def test_schema_validation_rule_keyerror_respects_max_errors():
    def bad_rule(df):
        _ = df["missing_column"]
        return []

    frame = ar.read_csv(io.StringIO("a\n1\n"))

    schema = ar.Schema(
        {
            "a": ar.String(),
        },
        rules=[bad_rule],
    )

    result = ar.validate(frame, schema, max_errors=1)

    assert result.issue_count == 1


def test_schema_validation_strict_max_errors_cap(tmp_path):
    path = tmp_path / "data.csv"

    path.write_text("name,extra1,extra2\njohn,a,b\n")

    frame = ar.read_csv(path)

    schema = ar.Schema(
        {
            "name": ar.String(),
        },
        strict=True,
    )

    result = ar.validate(frame, schema, max_errors=1)

    assert result.issue_count == 1
    assert len(result.issues) == 1


def test_schema_validation_unique_max_errors_cap(tmp_path):
    path = tmp_path / "data.csv"

    path.write_text("id\n1\n1\n1\n")

    frame = ar.read_csv(path)

    schema = ar.Schema(
        {
            "id": ar.Int64(),
        },
        unique=["id"],
    )

    result = ar.validate(frame, schema, max_errors=1)

    assert result.issue_count == 1
    assert len(result.issues) == 1


def test_validation_result_summary_counts_issues_across_multiple_columns():
    result = ar.ValidationResult(
        row_count=3,
        issue_count=4,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=0
            ),
            ar.ValidationIssue(
                column="status", rule="allowed", message="bad status", row_index=1
            ),
            ar.ValidationIssue(
                column="email", rule="email", message="bad email", row_index=1
            ),
            ar.ValidationIssue(
                column=None, rule="required_column", message="missing column"
            ),
        ],
        bad_rows=[0, 1],
    )

    summary = result.summary()

    assert summary["issues_by_rule"] == {
        "min": 1,
        "allowed": 1,
        "email": 1,
        "required_column": 1,
    }
    assert summary["issues_by_column"] == {"age": 1, "status": 1, "email": 1}
    assert summary["issues_by_column_and_rule"] == {
        "age": {"min": 1},
        "status": {"allowed": 1},
        "email": {"email": 1},
    }


def test_validation_result_summary_counts_grouped_rules_under_one_column():
    result = ar.ValidationResult(
        row_count=2,
        issue_count=3,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=0
            ),
            ar.ValidationIssue(
                column="age", rule="max", message="too large", row_index=1
            ),
            ar.ValidationIssue(
                column="age", rule="numeric", message="not numeric", row_index=1
            ),
        ],
        bad_rows=[0, 1],
    )

    summary = result.summary()

    assert summary["issues_by_rule"] == {"min": 1, "max": 1, "numeric": 1}
    assert summary["issues_by_column"] == {"age": 3}
    assert summary["issues_by_column_and_rule"] == {
        "age": {"min": 1, "max": 1, "numeric": 1}
    }


def test_validation_result_summary_counts_no_issue_result():
    result = ar.ValidationResult(row_count=3, issue_count=0, issues=[], bad_rows=[])

    summary = result.summary()

    assert summary["passed"] is True
    assert summary["issue_count"] == 0
    assert summary["bad_row_count"] == 0
    assert summary["issues_by_rule"] == {}
    assert summary["issues_by_column"] == {}
    assert summary["issues_by_column_and_rule"] == {}


def test_validation_result_summary_severity_counts_error():
    """severity_counts must be populated for issues with default 'error' severity."""
    result = ar.ValidationResult(
        row_count=3,
        issue_count=2,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=0
            ),
            ar.ValidationIssue(
                column="name", rule="max", message="too long", row_index=1
            ),
        ],
        bad_rows=[0, 1],
    )
    summary = result.summary()
    assert summary["severity_counts"] == {"error": 2}


def test_validation_result_summary_severity_counts_mixed():
    """severity_counts must track different severity levels."""
    result = ar.ValidationResult(
        row_count=5,
        issue_count=4,
        issues=[
            ar.ValidationIssue(
                column="x", rule="min", message="small", row_index=0, severity="error"
            ),
            ar.ValidationIssue(
                column="x", rule="max", message="large", row_index=1, severity="warning"
            ),
            ar.ValidationIssue(
                column="x",
                rule="required",
                message="missing",
                row_index=2,
                severity="error",
            ),
            ar.ValidationIssue(
                column="x",
                rule="nullable",
                message="null",
                row_index=3,
                severity="warning",
            ),
        ],
        bad_rows=[0, 1, 2, 3],
    )
    summary = result.summary()
    assert summary["severity_counts"] == {"error": 2, "warning": 2}


def test_validation_result_summary_issue_count_field():
    """summary issue_count must match the result's issue_count field."""
    result = ar.ValidationResult(
        row_count=10,
        issue_count=5,
        issues=[
            ar.ValidationIssue(column="a", rule="min", message="bad", row_index=i)
            for i in range(5)
        ],
        bad_rows=list(range(5)),
    )
    summary = result.summary()
    assert summary["issue_count"] == 5
    assert summary["passed"] is False


def test_validation_result_summary_bad_row_count():
    """summary bad_row_count must equal len(bad_rows)."""
    result = ar.ValidationResult(
        row_count=7,
        issue_count=3,
        issues=[
            ar.ValidationIssue(column="a", rule="min", message="bad", row_index=i)
            for i in [1, 3, 5]
        ],
        bad_rows=[1, 3, 5],
    )
    summary = result.summary()
    assert summary["bad_row_count"] == 3


def test_validation_result_summary_no_issues_severity_counts_empty():
    """When there are no issues, severity_counts must be an empty dict."""
    result = ar.ValidationResult(row_count=3, issue_count=0, issues=[], bad_rows=[])
    summary = result.summary()
    assert summary["severity_counts"] == {}


def test_schema_diff_summary_differences_by_change():
    """SchemaDiff.summary() differences_by_change must aggregate by change kind."""
    diff = ar.SchemaDiff(
        [
            ar.SchemaDiffEntry(
                change="added_column",
                column="new_col",
            ),
            ar.SchemaDiffEntry(
                change="changed_field",
                column="id",
            ),
            ar.SchemaDiffEntry(
                change="added_column",
                column="another_col",
            ),
        ],
    )
    summary = diff.summary()
    assert summary["differences_by_change"] == {"added_column": 2, "changed_field": 1}


def test_schema_diff_summary_differences_by_column():
    """SchemaDiff.summary() differences_by_column must aggregate by column name."""
    diff = ar.SchemaDiff(
        [
            ar.SchemaDiffEntry(
                change="removed_column",
                column="x",
            ),
            ar.SchemaDiffEntry(
                change="changed_type",
                column="x",
            ),
            ar.SchemaDiffEntry(
                change="added_column",
                column="y",
            ),
        ],
    )
    summary = diff.summary()
    assert summary["differences_by_column"] == {"x": 2, "y": 1}


def test_schema_diff_summary_no_differences():
    """SchemaDiff.summary() with no differences must return empty aggregations."""
    diff = ar.SchemaDiff([])
    summary = diff.summary()
    assert summary["changed"] is False
    assert summary["difference_count"] == 0
    assert summary["differences_by_change"] == {}
    assert summary["differences_by_column"] == {}


def test_validation_result_to_pandas(sample_csv):
    result = ar.validate(
        ar.read_csv(sample_csv),
        {"age": ar.Int64(min=31)},
    )
    df = result.to_pandas()
    assert list(df["rule"]) == ["min", "min"]
    assert list(df["row_index"]) == [1, 2]


def test_validation_result_to_markdown_for_success(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64()})

    markdown = result.to_markdown()

    assert "## Validation Report" in markdown
    assert "- Status: **passed**" in markdown
    assert "- Issues found: 0" in markdown
    assert "| Column | Rule | Row | Value | Message |" not in markdown


def test_warning_severity_does_not_fail_validation(tmp_path):
    path = tmp_path / "warnings.csv"
    path.write_text("age\n15\n")

    schema = {
        "age": ar.Field(
            dtype="int64",
            min=18,
            severity="warning",
        )
    }

    result = ar.validate(ar.read_csv(path), schema)

    assert result.passed
    assert result.issue_count == 1
    assert result.issues[0].severity == "warning"
    assert result.issues[0].rule == "min"


def test_warning_severity_does_not_fail_dtype_mismatch(tmp_path):
    path = tmp_path / "dtype_warning.csv"
    path.write_text("age\nhello\n")

    result = ar.validate(
        ar.read_csv(path),
        {"age": ar.Int64(severity="warning")},
    )

    assert result.passed
    assert result.issue_count == 1
    assert result.issues[0].rule == "dtype"
    assert result.issues[0].severity == "warning"


def test_validation_result_to_markdown_includes_issue_table(sample_csv):
    result = ar.validate(
        ar.read_csv(sample_csv),
        {"age": ar.Int64(min=31), "missing": ar.String()},
    )

    # Default: redact_values=False — raw values are shown
    markdown = result.to_markdown()

    assert "- Status: **failed**" in markdown
    assert "- Issues found: 3" in markdown
    assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
    assert "| age | min | error | 1 |" in markdown
    assert (
        "| missing | required_column | error |  |  | Missing required column: missing |"
        in markdown
    )


def test_validation_result_to_markdown_limits_visible_issues(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64(min=31)})

    markdown = result.to_markdown(max_issues=1)

    assert "| age | min | error | 1 |" in markdown
    assert "| age | min | 2 |" not in markdown
    assert "_Showing 1 of 2 issues._" in markdown


def test_validation_result_to_markdown_escapes_table_cells():
    result = ar.ValidationResult(
        row_count=1,
        issue_count=1,
        issues=[
            ar.ValidationIssue(
                column="notes|raw",
                rule="pattern",
                row_index=0,
                value="left|right\nnext",
                message="Expected one|two\nlines",
            )
        ],
        bad_rows=[0],
    )

    # Column, value, and message cells are escaped (default: redact_values=False)
    markdown = result.to_markdown()
    assert "notes\\|raw" in markdown
    assert "left\\|right<br>next" in markdown
    assert "Expected one\\|two<br>lines" in markdown

    # Opt-in to redaction — value is replaced with [REDACTED]
    markdown_redacted = result.to_markdown(redact_values=True)
    assert "notes\\|raw" in markdown_redacted
    assert "[REDACTED]" in markdown_redacted
    assert "Expected one\\|two<br>lines" in markdown_redacted


def test_validation_result_to_markdown_rejects_negative_max_issues(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64(min=31)})

    try:
        result.to_markdown(max_issues=-1)
    except ValueError as exc:
        assert "max_issues" in str(exc)
    else:
        raise AssertionError("Expected max_issues validation to raise")


def test_validation_result_to_markdown_rejects_non_integer_max_issues(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64(min=31)})

    for invalid in ("1", 1.5, True):
        try:
            result.to_markdown(max_issues=invalid)  # type: ignore[arg-type]
        except TypeError as exc:
            assert "max_issues must be an integer or None" in str(exc)
        else:
            raise AssertionError(f"Expected max_issues={invalid!r} to raise")


def test_schema_construction_validates_rules():
    with pytest.raises(TypeError, match="Schema 'rules' must be a list of callables"):
        ar.Schema({"x": ar.Int64()}, rules="abc")

    with pytest.raises(TypeError, match="Schema 'rules' must be a list of callables"):
        ar.Schema({"x": ar.Int64()}, rules=123)

    with pytest.raises(TypeError, match="Schema 'rules' must be a list of callables"):
        ar.Schema({"x": ar.Int64()}, rules=object())

    with pytest.raises(TypeError, match="Schema 'rules' must be a list of callables"):
        ar.Schema({"x": ar.Int64()}, rules=[object()])

    def valid_rule(df):
        return []

    with pytest.raises(TypeError, match="Schema 'rules' must be a list of callables"):
        ar.Schema({"x": ar.Int64()}, rules=[valid_rule, 456])

    assert ar.Schema({"x": ar.Int64()}, rules=[valid_rule]).rules is not None
    assert ar.Schema({"x": ar.Int64()}, rules=(valid_rule,)).rules is not None
    assert ar.Schema({"x": ar.Int64()}, rules=None).rules is None


# ---------------------------------------------------------------------------
# Regression tests: redaction policy for ValidationResult.to_markdown
# ---------------------------------------------------------------------------


def test_validation_result_to_markdown_does_not_redact_by_default():
    """Value column must contain raw value when redact_values=False (default)."""
    result = ar.ValidationResult(
        row_count=1,
        issue_count=1,
        issues=[
            ar.ValidationIssue(
                column="email",
                rule="email",
                row_index=1,
                value="secret@internal.example.com",
                message="Invalid email",
            )
        ],
        bad_rows=[1],
    )

    markdown = result.to_markdown()  # default: redact_values=False

    assert "[REDACTED]" not in markdown
    assert "secret@internal.example.com" in markdown


def test_validation_result_to_markdown_redacts_when_opted_in():
    """Value column must contain [REDACTED] when redact_values=True."""
    result = ar.ValidationResult(
        row_count=1,
        issue_count=1,
        issues=[
            ar.ValidationIssue(
                column="email",
                rule="email",
                row_index=1,
                value="secret@internal.example.com",
                message="Invalid email",
            )
        ],
        bad_rows=[1],
    )

    markdown = result.to_markdown(redact_values=True)

    assert "[REDACTED]" in markdown
    assert "secret@internal.example.com" not in markdown


def test_validation_result_to_markdown_redacted_output_is_deterministic():
    """to_markdown() must return identical output on repeated calls."""
    result = ar.ValidationResult(
        row_count=2,
        issue_count=2,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", row_index=1, value=-5, message="below 0"
            ),
            ar.ValidationIssue(
                column="age", rule="max", row_index=2, value=999, message="above 120"
            ),
        ],
        bad_rows=[1, 2],
    )

    assert result.to_markdown() == result.to_markdown()
    assert result.to_markdown(redact_values=True) == result.to_markdown(
        redact_values=True
    )


def test_validation_result_to_markdown_none_value_redacted():
    """None/missing values are also replaced with [REDACTED] when redaction is enabled."""
    result = ar.ValidationResult(
        row_count=1,
        issue_count=1,
        issues=[
            ar.ValidationIssue(
                column="col",
                rule="nullable",
                row_index=1,
                value=None,
                message="Null not allowed",
            )
        ],
        bad_rows=[1],
    )

    markdown = result.to_markdown(redact_values=True)  # explicit redaction
    assert "[REDACTED]" in markdown

    markdown_raw = result.to_markdown()  # default redaction is False
    # None -> empty cell in raw mode
    assert "[REDACTED]" not in markdown_raw


def _make_failing_result() -> ar.ValidationResult:
    """Helper: a ValidationResult with one issue, for redact_values type tests."""
    return ar.ValidationResult(
        row_count=1,
        issue_count=1,
        issues=[
            ar.ValidationIssue(
                column="col",
                rule="min",
                row_index=1,
                value=0,
                message="below minimum",
            )
        ],
        bad_rows=[1],
    )


def test_to_markdown_rejects_non_bool_redact_values():
    """to_markdown() must raise TypeError for any non-bool redact_values argument."""
    result = _make_failing_result()

    for invalid in ("false", "true", "", 0, 1, None, [], {}):
        try:
            result.to_markdown(redact_values=invalid)  # type: ignore[arg-type]
        except TypeError as exc:
            assert "redact_values must be a bool" in str(
                exc
            ), f"Wrong error message for {invalid!r}: {exc}"
        else:
            raise AssertionError(
                f"Expected TypeError for redact_values={invalid!r}, but no exception was raised"
            )


def test_to_markdown_accepts_bool_redact_values():
    """to_markdown() must not raise for redact_values=True or redact_values=False."""
    result = _make_failing_result()

    md_false = result.to_markdown(redact_values=False)
    md_true = result.to_markdown(redact_values=True)

    assert "0" in md_false, "Raw value should appear when redact_values=False"
    assert "[REDACTED]" in md_true, "[REDACTED] should appear when redact_values=True"
    assert "0" not in md_true.split("| Value |")[-1] or "[REDACTED]" in md_true


def test_unique_constraint_detects_duplicates(tmp_path):
    path = tmp_path / "unique.csv"
    path.write_text("id,value\n1,100\n2,200\n1,300\n3,400\n")
    result = ar.validate(ar.read_csv(path), {"id": ar.Int64(unique=True)})
    assert not result.passed
    assert any(
        issue.rule == "unique" and issue.column == "id" for issue in result.issues
    )


def test_validation_result_summary_counts_repeated_issues_in_one_column():
    result = ar.ValidationResult(
        row_count=3,
        issue_count=3,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=0
            ),
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=1
            ),
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=2
            ),
        ],
        bad_rows=[0, 1, 2],
    )

    summary = result.summary()

    assert summary["issues_by_rule"] == {"min": 3}
    assert summary["issues_by_column"] == {"age": 3}
    assert summary["issues_by_column_and_rule"] == {"age": {"min": 3}}


def test_validation_result_summary_counts_issues_across_multiple_columns():
    result = ar.ValidationResult(
        row_count=3,
        issue_count=4,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=0
            ),
            ar.ValidationIssue(
                column="status", rule="allowed", message="bad status", row_index=1
            ),
            ar.ValidationIssue(
                column="email", rule="email", message="bad email", row_index=1
            ),
            ar.ValidationIssue(
                column=None, rule="required_column", message="missing column"
            ),
        ],
        bad_rows=[0, 1],
    )

    summary = result.summary()

    assert summary["issues_by_rule"] == {
        "min": 1,
        "allowed": 1,
        "email": 1,
        "required_column": 1,
    }
    assert summary["issues_by_column"] == {"age": 1, "status": 1, "email": 1}
    assert summary["issues_by_column_and_rule"] == {
        "age": {"min": 1},
        "status": {"allowed": 1},
        "email": {"email": 1},
    }


def test_validation_result_summary_counts_grouped_rules_under_one_column():
    result = ar.ValidationResult(
        row_count=2,
        issue_count=3,
        issues=[
            ar.ValidationIssue(
                column="age", rule="min", message="too small", row_index=0
            ),
            ar.ValidationIssue(
                column="age", rule="max", message="too large", row_index=1
            ),
            ar.ValidationIssue(
                column="age", rule="numeric", message="not numeric", row_index=1
            ),
        ],
        bad_rows=[0, 1],
    )

    summary = result.summary()

    assert summary["issues_by_rule"] == {"min": 1, "max": 1, "numeric": 1}
    assert summary["issues_by_column"] == {"age": 3}
    assert summary["issues_by_column_and_rule"] == {
        "age": {"min": 1, "max": 1, "numeric": 1}
    }


def test_validation_result_summary_counts_no_issue_result():
    result = ar.ValidationResult(row_count=3, issue_count=0, issues=[], bad_rows=[])

    summary = result.summary()

    assert summary["passed"] is True
    assert summary["issue_count"] == 0
    assert summary["bad_row_count"] == 0
    assert summary["issues_by_rule"] == {}
    assert summary["issues_by_column"] == {}
    assert summary["issues_by_column_and_rule"] == {}


def test_validation_result_to_markdown_for_success(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64()})

    markdown = result.to_markdown()

    assert "## Validation Report" in markdown
    assert "- Status: **passed**" in markdown
    assert "- Issues found: 0" in markdown
    assert "| Column | Rule | Row | Value | Message |" not in markdown


def test_validation_result_to_markdown_includes_issue_table(sample_csv):
    result = ar.validate(
        ar.read_csv(sample_csv),
        {"age": ar.Int64(min=31), "missing": ar.String()},
    )

    markdown = result.to_markdown()

    assert "- Status: **failed**" in markdown
    assert "- Issues found: 3" in markdown
    assert "| Column | Rule | Row | Value | Message |" in markdown
    assert "| age | min | 0 |" in markdown
    assert (
        "| missing | required_column |  |  | Missing required column: missing |"
        in markdown
    )


def test_validation_result_to_markdown_limits_visible_issues(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64(min=31)})

    markdown = result.to_markdown(max_issues=1)

    assert "| age | min | 0 |" in markdown
    assert "| age | min | 1 |" not in markdown
    assert "_Showing 1 of 2 issues._" in markdown


def test_validation_result_to_markdown_escapes_table_cells():
    result = ar.ValidationResult(
        row_count=1,
        issue_count=1,
        issues=[
            ar.ValidationIssue(
                column="notes|raw",
                rule="pattern",
                row_index=0,
                value="left|right\nnext",
                message="Expected one|two\nlines",
            )
        ],
        bad_rows=[0],
    )

    markdown = result.to_markdown()

    assert "notes\\|raw" in markdown
    assert "left\\|right<br>next" in markdown
    assert "Expected one\\|two<br>lines" in markdown


def test_validation_result_to_markdown_rejects_negative_max_issues(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64(min=31)})

    try:
        result.to_markdown(max_issues=-1)
    except ValueError as exc:
        assert "max_issues" in str(exc)
    else:
        raise AssertionError("Expected max_issues validation to raise")


def test_validation_result_to_markdown_rejects_non_integer_max_issues(sample_csv):
    result = ar.validate(ar.read_csv(sample_csv), {"age": ar.Int64(min=31)})

    for invalid in ("1", 1.5, True):
        try:
            result.to_markdown(max_issues=invalid)  # type: ignore[arg-type]
        except TypeError as exc:
            assert "max_issues must be an integer or None" in str(exc)
        else:
            raise AssertionError(f"Expected max_issues={invalid!r} to raise")


def test_custom_pattern_validation(tmp_path):
    path = tmp_path / "codes.csv"
    path.write_text("code\nAA-123\nbad\n")
    result = ar.validate(
        ar.read_csv(path), {"code": ar.String(pattern=r"^[A-Z]{2}-\d{3}$")}
    )

    assert not result.passed
    assert result.issues[0].rule == "pattern"
    assert result.issues[0].row_index == 2


def test_row_index_is_one_based_for_first_row(tmp_path):
    path = tmp_path / "codes.csv"
    path.write_text("age\n-1\n30\n25\n")
    frame = ar.read_csv(path)
    result = ar.validate(frame, {"age": ar.Int64(min=0)})

    assert not result.passed
    assert len(result.issues) == 1
    assert result.issues[0].row_index == 1


def test_compare_schema_method(sample_csv, tmp_path):
    # 1. Base Frame and Matching Frame Setup
    df_base = ar.read_csv(sample_csv)
    df_match = ar.read_csv(sample_csv)

    # 2. Setup Shuffled/Swapped Order Frame
    shuffled_path = tmp_path / "shuffled.csv"
    shuffled_path.write_text("age,name,email,active\n" "30,Alice,alice@test.com,True\n")
    df_shuffled = ar.read_csv(shuffled_path)

    # 3. Setup Wrong Data Type Frame
    wrong_dtype_path = tmp_path / "wrong_dtype.csv"
    wrong_dtype_path.write_text(
        "name,age,email,active\n" "Alice,30.5,alice@test.com,True\n"
    )
    df_wrong_dtype = ar.read_csv(wrong_dtype_path)

    # 4. Setup Wrong Column Names Frame
    wrong_cols_path = tmp_path / "wrong_cols.csv"
    wrong_cols_path.write_text(
        "name,age,email,status\n" "Alice,30,alice@test.com,active\n"
    )
    df_wrong_cols = ar.read_csv(wrong_cols_path)

    # --- ASSERTIONS ---
    # Requirement A: Same schema test
    assert df_base.compare_schema(df_match, strict=True) is True
    assert df_base.compare_schema(df_match, strict=False) is True

    # Requirement B: Strict vs Non-Strict order behavior tracking
    assert df_base.compare_schema(df_shuffled, strict=True) is False
    assert df_base.compare_schema(df_shuffled, strict=False) is True

    # Requirement C: Data type mismatch validation
    assert df_base.compare_schema(df_wrong_dtype, strict=False) is False

    # Requirement D: Column naming structural mismatch validation
    assert df_base.compare_schema(df_wrong_cols, strict=False) is False

    # Requirement E: Invalid object class input safe rejection handling
    with pytest.raises(TypeError):
        df_base.compare_schema(["not", "an", "ArFrame", "object"])


def test_string_min_length_boundary(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("name\nab\nabc\n")

    result = ar.validate(
        ar.read_csv(path),
        {"name": ar.String(min_length=3)},
    )

    assert not result.passed
    assert result.issue_count == 1
    assert result.issues[0].rule == "min_length"
    assert result.issues[0].row_index == 0


def test_string_max_length_boundary(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("name\nabcde\nabcdef\n")

    result = ar.validate(
        ar.read_csv(path),
        {"name": ar.String(max_length=5)},
    )

    assert not result.passed
    assert result.issue_count == 1
    assert result.issues[0].rule == "max_length"
    assert result.issues[0].row_index == 1


def test_null_values_skip_length_validation(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("name\n\nabcd\n")

    result = ar.validate(
        ar.read_csv(path),
        {"name": ar.String(min_length=5)},
    )

    assert not result.passed
    assert result.issue_count == 1
    assert result.issues[0].rule == "min_length"
    assert result.issues[0].row_index == 0


def test_compare_schema_matching(sample_csv):
    """Test that identical schemas match under both strict and non-strict modes."""
    df_base = ar.read_csv(sample_csv)
    df_match = ar.read_csv(sample_csv)

    assert df_base.compare_schema(df_match, strict=True) is True
    assert df_base.compare_schema(df_match, strict=False) is True


def test_compare_schema_order_difference(sample_csv, tmp_path):
    """Test that column order differences fail strict mode but pass non-strict mode."""
    df_base = ar.read_csv(sample_csv)

    shuffled_path = tmp_path / "shuffled.csv"
    shuffled_path.write_text("age,name,email,active\n" "30,Alice,alice@test.com,True\n")
    df_shuffled = ar.read_csv(shuffled_path)

    assert df_base.compare_schema(df_shuffled, strict=True) is False
    assert df_base.compare_schema(df_shuffled, strict=False) is True


def test_compare_schema_dtype_mismatch(sample_csv, tmp_path):
    """Test that schema matching fails when column data types mismatch."""
    df_base = ar.read_csv(sample_csv)

    wrong_dtype_path = tmp_path / "wrong_dtype.csv"
    wrong_dtype_path.write_text(
        "name,age,email,active\n" "Alice,30.5,alice@test.com,True\n"
    )
    df_wrong_dtype = ar.read_csv(wrong_dtype_path)

    assert df_base.compare_schema(df_wrong_dtype, strict=False) is False


def test_compare_schema_column_mismatch(sample_csv, tmp_path):
    """Test that schema matching fails when column names do not match."""
    df_base = ar.read_csv(sample_csv)

    wrong_cols_path = tmp_path / "wrong_cols.csv"
    wrong_cols_path.write_text(
        "name,age,email,status\n" "Alice,30,alice@test.com,active\n"
    )
    df_wrong_cols = ar.read_csv(wrong_cols_path)

    assert df_base.compare_schema(df_wrong_cols, strict=False) is False


def test_compare_schema_invalid_input(sample_csv):
    """Test that passing a non-ArFrame object correctly raises a TypeError."""
    df_base = ar.read_csv(sample_csv)

    with pytest.raises(TypeError):
        df_base.compare_schema(["not", "an", "ArFrame", "object"])
