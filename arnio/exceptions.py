"""arnio.exceptions — Exception hierarchy for Arnio.

Design rules:
    - Validation NEVER raises on bad data — it returns structured results.
    - Usage errors (wrong types, invalid schemas, unsupported data) DO raise.
    - Every exception message explains what went wrong, why, and how to fix it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arnio.validate._result import Issue


class ArnioError(Exception):
    """Base exception for all Arnio errors."""


class SchemaError(ArnioError):
    """Raised when a schema definition is invalid.

    Examples of triggers:
        - Field type used with incompatible parameters
        - min > max in a numeric field
        - Invalid regex pattern
        - Unknown field type in deserialized schema
    """


class AdapterError(ArnioError):
    """Raised when the input data type is not supported.

    This means Arnio could not find an adapter for the given data.
    Includes a message showing what was received and what types are supported.
    """

    def __init__(self, data: Any) -> None:
        type_name = type(data).__qualname__
        super().__init__(
            f"Unsupported data type: {type_name!r}.\n"
            f"Arnio supports: pandas DataFrame, list[dict], dict[str, list].\n"
            f"To add support for custom types, implement the DataFrameAdapter protocol."
        )


class ValidationError(ArnioError):
    """Raised by ``ar.check()`` when data fails validation.

    Carries structured issue data so callers can inspect failures
    programmatically in CI or assertion contexts.

    Attributes:
        issues: List of validation issues that caused the failure.
    """

    def __init__(self, message: str, *, issues: list[Issue] | None = None) -> None:
        self.issues: list[Issue] = issues or []
        super().__init__(message)


class CleaningError(ArnioError):
    """Raised when a cleaning step encounters an unrecoverable error.

    Attributes:
        step_name: The name of the cleaning step that failed.
    """

    def __init__(self, step_name: str, detail: str) -> None:
        self.step_name = step_name
        super().__init__(
            f"Cleaning step {step_name!r} failed: {detail}"
        )


class PipelineError(ArnioError):
    """Raised when a pipeline encounters an error during execution.

    Attributes:
        step_name: The name of the step that failed.
        step_index: The 0-based index of the failing step in the pipeline.
    """

    def __init__(
        self, step_name: str, step_index: int, cause: Exception
    ) -> None:
        self.step_name = step_name
        self.step_index = step_index
        self.__cause__ = cause
        super().__init__(
            f"Pipeline failed at step {step_index} ({step_name!r}): {cause}"
        )
