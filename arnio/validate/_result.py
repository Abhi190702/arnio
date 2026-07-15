"""Validation result types — Issue and ValidationResult."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Issue:
    """A single validation issue found during schema validation.

    Attributes:
        column: Column name where the issue was found.
        rule: The validation rule that was violated.
        message: Human-readable description of the issue.
        severity: "error" or "warning".
        row_index: Row index where the issue was found (None for column-level issues).
        value: The offending value (None for column-level issues).
    """

    column: str
    rule: str
    message: str
    severity: str = "error"
    row_index: int | None = None
    value: Any = None


@dataclass
class ValidationResult:
    """Result of validating data against a schema.

    Contains all issues found during validation. The validation engine
    NEVER raises on bad data — it always returns a ValidationResult.

    Attributes:
        issues: All validation issues found.
        schema_name: Optional name of the schema used (for display).
    """

    issues: list[Issue] = field(default_factory=list)
    schema_name: str | None = None

    @property
    def passed(self) -> bool:
        """True if there are no error-level issues."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def issue_count(self) -> int:
        """Total number of issues."""
        return len(self.issues)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == "warning")

    def __bool__(self) -> bool:
        """A ValidationResult is truthy if validation passed (no errors)."""
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {
            "passed": self.passed,
            "issue_count": self.issue_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "column": i.column,
                    "rule": i.rule,
                    "message": i.message,
                    "severity": i.severity,
                    "row_index": i.row_index,
                    "value": _serialize_value(i.value),
                }
                for i in self.issues
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_pandas(self) -> Any:
        """Convert issues to a pandas DataFrame.

        Returns:
            A ``pd.DataFrame`` with one row per issue.
        """
        import pandas as pd

        if not self.issues:
            return pd.DataFrame(
                columns=["column", "rule", "message", "severity", "row_index", "value"]
            )
        return pd.DataFrame([
            {
                "column": i.column,
                "rule": i.rule,
                "message": i.message,
                "severity": i.severity,
                "row_index": i.row_index,
                "value": i.value,
            }
            for i in self.issues
        ])

    def to_markdown(self) -> str:
        """Render issues as a Markdown table."""
        if not self.issues:
            return "✅ **Validation passed** — no issues found.\n"

        lines = [
            f"{'❌' if not self.passed else '⚠️'} **Validation {'failed' if not self.passed else 'passed with warnings'}** "
            f"— {self.issue_count} issue(s) found.\n",
            "| Column | Rule | Message | Severity |",
            "|--------|------|---------|----------|",
        ]
        for i in self.issues:
            lines.append(f"| {i.column} | {i.rule} | {i.message} | {i.severity} |")
        return "\n".join(lines) + "\n"

    def to_html(self) -> str:
        """Render issues as an HTML report."""
        if not self.issues:
            return '<div class="arnio-validation passed"><h3>✅ Validation Passed</h3></div>'

        status = "failed" if not self.passed else "warnings"
        rows = "\n".join(
            f"<tr><td>{i.column}</td><td>{i.rule}</td>"
            f"<td>{i.message}</td><td>{i.severity}</td></tr>"
            for i in self.issues
        )
        return (
            f'<div class="arnio-validation {status}">'
            f"<h3>{'❌' if not self.passed else '⚠️'} Validation "
            f"{'Failed' if not self.passed else 'Passed with Warnings'}</h3>"
            f"<p>{self.issue_count} issue(s) found</p>"
            f"<table><thead><tr>"
            f"<th>Column</th><th>Rule</th><th>Message</th><th>Severity</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    def _repr_html_(self) -> str:
        """Jupyter notebook rendering."""
        return self.to_html()

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ValidationResult({status}, "
            f"errors={self.error_count}, warnings={self.warning_count})"
        )


def _serialize_value(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
