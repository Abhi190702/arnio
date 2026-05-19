"""
arnio.convert
Pandas conversion functions.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pyarrow as pa

from ._core import _DType, _Frame
from .frame import ArFrame


def _is_nested(value: object) -> bool:
    return isinstance(value, (list, dict, tuple, set, np.ndarray))


def _to_binding_safe(value: Any) -> Any:
    """
    Internal helper that normalizes scalars for the C++ binding layer.

    Parameters
    ----------
    value : Any
        Input value to convert.

    Returns
    -------
    Any
        Value safe for C++ binding. Decimal inputs are preserved as exact
        strings. Float inputs are converted to binary float. NaN/Infinity are
        rejected.

    Raises
    ------
    ValueError
        If the value is NaN or infinite.
    """
    if isinstance(value, decimal.Decimal):
        if value.is_nan() or value.is_infinite():
            raise ValueError("Invalid financial value: NaN or Infinity.")
        return str(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Invalid financial value: NaN or Infinity.")
        return float(value)

    return value


def _check_unsupported_dtype(col_name: object, series: pd.Series) -> None:
    """Raise a clear TypeError for dtypes that arnio cannot convert."""
    dtype = series.dtype
    dtype_str = str(dtype)
    name = repr(str(col_name))

    if hasattr(dtype, "tz") or dtype_str.startswith("datetime64"):
        raise TypeError(
            f"Column {name} has unsupported dtype '{dtype_str}'.\n"
            f"  Fix: df[{name}] = df[{name}].astype(str)  "
            f"# or use .dt.strftime('%Y-%m-%d') for formatted dates"
        )

    if dtype_str.startswith("timedelta"):
        raise TypeError(
            f"Column {name} has unsupported dtype '{dtype_str}'.\n"
            f"  Fix: df[{name}] = df[{name}].dt.total_seconds()"
        )

    if hasattr(dtype, "categories"):
        raise TypeError(
            f"Column {name} has unsupported dtype 'category'.\n"
            f"  Fix: df[{name}] = df[{name}].astype(str)"
        )

    if dtype_str in ("complex128", "complex64"):
        raise TypeError(
            f"Column {name} has unsupported dtype '{dtype_str}'.\n"
            f"  Fix: df[{name}] = df[{name}].apply(str)"
        )


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def _normalize_scalar(value: object) -> object:
    if isinstance(value, decimal.Decimal):
        return _to_binding_safe(value)
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int) and not isinstance(value, bool):
        if value < -9223372036854775808 or value > 9223372036854775807:
            raise ValueError(
                f"Integer value {value} is out of bounds for signed 64-bit integer. "
                "arnio only supports signed 64-bit integers (-9223372036854775808 to 9223372036854775807)."
            )
    if isinstance(value, float):
        return _to_binding_safe(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < INT64_MIN or value > INT64_MAX:
            raise ValueError(
                f"Integer value {value!r} is outside the signed int64 range "
                f"[{INT64_MIN}, {INT64_MAX}]. "
                "Convert the column to string first: df[col] = df[col].astype(str)"
            )
        return value
    if not isinstance(value, str):
        return str(value)
    return value


def _scalar_kind(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "string"


def _series_to_python_values(series: pd.Series, col_name: object) -> list[object]:
    values: list[object] = []
    kinds: set[str] = set()

    _ALLOWED_SCALAR_TYPES = (str, int, float, bool, decimal.Decimal)

    for raw in series.tolist():
        if _is_nested(raw):
            raise TypeError(
                f"Column '{col_name}' contains unsupported nested value "
                f"of type '{type(raw).__name__}' at value {raw!r}. "
                "Convert nested objects to strings or flatten them first."
            )
        try:
            value = _normalize_scalar(raw)
        except ValueError as e:
            raise ValueError(f"Column '{col_name}': {e}") from e

        values.append(value)
        if value is not None:
            kinds.add(_scalar_kind(value))

    if "string" in kinds and len(kinds) > 1:
        return [None if value is None else str(value) for value in values]

    if "bool" in kinds and len(kinds) > 1:
        return [None if value is None else str(value) for value in values]

    if kinds == {"int", "float"}:
        return [None if value is None else float(value) for value in values]

    return values


def to_pandas(frame: ArFrame, *, copy: bool = False) -> pd.DataFrame:
    """Convert ArFrame to pandas.DataFrame.

    Parameters
    ----------
    frame : ArFrame
        Input ArFrame to convert.
    copy : bool, default False
        When False, preserve the fast zero-copy path where supported. Some
        columns still require copies because of null-mask handling, Python
        object creation, or binding limitations. When True, return defensive
        pandas-owned copies of supported column buffers.

    Returns
    -------
    pd.DataFrame
        Equivalent pandas DataFrame with proper dtypes and null handling.
        If the ArFrame was created via ``from_pandas()``, any ``attrs``
        metadata from the original DataFrame is restored on the result.

    Examples
    --------
    >>> frame = ar.read_csv("data.csv")
    >>> df = ar.to_pandas(frame)
    >>> defensive_df = ar.to_pandas(frame, copy=True)
    """
    if not isinstance(copy, bool):
        raise TypeError("copy must be a bool")

    if not isinstance(frame, ArFrame):
        raise TypeError(
            f"to_pandas() expects an ArFrame, got {type(frame).__name__}. Use arnio.from_pandas() first."
        )

    cpp_frame = frame._frame
    data = {}

    for i in range(cpp_frame.num_cols()):
        col = cpp_frame.column_by_index(i)
        name = col.name()
        dtype = col.dtype()
        mask = col.get_null_mask()

        if dtype == _DType.INT64:
            arr = col.to_numpy_int()
            if copy:
                arr = arr.copy()
            series = pd.Series(arr, dtype=pd.Int64Dtype())
            series[mask] = pd.NA
            data[name] = series
        elif dtype == _DType.FLOAT64:
            arr = col.to_numpy_float()
            if copy or mask.any():
                arr = arr.copy()
            if mask.any():
                arr[mask] = np.nan
            data[name] = arr
        elif dtype == _DType.BOOL:
            arr = col.to_numpy_bool()
            if copy:
                arr = arr.copy()
            series = pd.Series(arr, dtype=pd.BooleanDtype())
            series[mask] = pd.NA
            data[name] = series
        else:
            values = col.to_python_list()
            series = pd.Series(values, dtype=pd.StringDtype())
            series[mask] = pd.NA
            data[name] = series

    result = pd.DataFrame(data)
    if frame._attrs:
        result.attrs = copy.deepcopy(frame._attrs)
    return result


def _pandas_dtype_to_arnio(dtype: object) -> _DType | None:
    if dtype == pd.Int64Dtype():
        return _DType.INT64
    if dtype == pd.Float64Dtype() or dtype == np.dtype("float64"):
        return _DType.FLOAT64

    if dtype == pd.BooleanDtype() or dtype == np.dtype("bool"):
        return _DType.BOOL
    if dtype == pd.StringDtype():
        return _DType.STRING
    # object dtype is intentionally left to value-based inference
    if dtype == pd.BooleanDtype() or str(dtype) == "bool":
        return _DType.BOOL

    return None


def from_pandas(df: pd.DataFrame) -> ArFrame:
    """Convert pandas.DataFrame to ArFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input pandas DataFrame to convert.

    Returns
    -------
    ArFrame
        Equivalent ArFrame with inferred types.

    Raises
    ------
    TypeError
        If DataFrame contains unsupported nested/complex types.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"name": ["Alice"], "age": [25]})
    >>> frame = ar.from_pandas(df)
    """
    _validate_unique_column_labels(df.columns)

    columns = {}
    dtype_hints = {}

    for col_name in df.columns:
        series = df[col_name]
        name = str(col_name)

        _check_unsupported_dtype(col_name, series)

        columns[name] = _series_to_python_values(series, col_name)

        dtype_hint = _pandas_dtype_to_arnio(series.dtype)
        if dtype_hint is not None:
            dtype_hints[name] = dtype_hint

    cpp_frame = _Frame.from_dict(columns, dtype_hints)
    return ArFrame(cpp_frame, attrs=copy.deepcopy(df.attrs))


_NUMERIC_DTYPES = {"int64", "float64", "bool"}


def to_numpy(
    frame: ArFrame,
    columns: list[str] | None = None,
    *,
    null_value: float = np.nan,
    allow_non_numeric: bool = False,
) -> np.ndarray:
    """Extract columns from an ArFrame as a 2-D NumPy ``float64`` array.

    This helper is designed to prepare clean numeric arrays from messy
    tabular data, making it ideal for feeding data directly into
    scikit-learn models, statistical functions, and other numerical workflows.

    Parameters
    ----------
    frame : ArFrame
        Source data.
    columns : list[str] or None
        Column names to include.  When *None* (the default), all numeric
        columns (``int64``, ``float64``, ``bool``) are auto-selected in
        their original order.
    null_value : float, optional
        Sentinel substituted for null / missing cells (default ``np.nan``).
    allow_non_numeric : bool, optional
        If *False* (default), selecting a non-numeric column raises
        ``TypeError``.  If *True*, non-numeric columns are silently
        skipped when *columns* is ``None``; explicitly requested
        non-numeric columns still raise ``TypeError``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_rows, n_selected_columns)`` with dtype
        ``float64``.  Row order matches the ArFrame exactly.

    Raises
    ------
    ValueError
        If any requested column name does not exist in the frame, or if
        no numeric columns are available when *columns* is ``None``.
    TypeError
        If a selected column has a non-numeric dtype and either
        *allow_non_numeric* is ``False`` or the column was explicitly
        requested via *columns*.

    Examples
    --------
    >>> frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))
    >>> ar.to_numpy(frame)
    array([[1., 3.],
           [2., 4.]])

    >>> ar.to_numpy(frame, columns=["b"])
    array([[3.],
           [4.]])
    """
    dtypes = frame.dtypes
    all_columns = frame.columns

    # --- validate columns input ---
    if columns is not None:
        if isinstance(columns, (str, bytes)):
            raise TypeError(
                "columns must be a list of column names, not a string. "
                "Use columns=['" + str(columns) + "'] instead."
            )
        if not isinstance(columns, (list, tuple)):
            raise TypeError(
                "columns must be a list of column names, not "
                f"{type(columns).__name__}"
            )

    # --- resolve column list ---
    if columns is not None:
        missing = [c for c in columns if c not in all_columns]
        if missing:
            raise ValueError(
                f"Unknown columns: {missing}. Available columns: {all_columns}"
            )
        selected = list(columns)
    else:
        selected = [c for c in all_columns if dtypes[c] in _NUMERIC_DTYPES]
        if not selected:
            raise ValueError(
                "No numeric columns found in frame. "
                "Pass explicit column names via the 'columns' parameter."
            )

    # --- validate dtypes ---
    non_numeric = [c for c in selected if dtypes[c] not in _NUMERIC_DTYPES]
    if non_numeric:
        if columns is not None:
            # Explicitly requested non-numeric columns always raise,
            # regardless of allow_non_numeric.
            raise TypeError(
                f"Non-numeric columns selected: {non_numeric}. "
                "Remove them from the 'columns' list."
            )
        if not allow_non_numeric:
            raise TypeError(
                f"Non-numeric columns selected: {non_numeric}. "
                "Set allow_non_numeric=True to skip them, or remove them "
                "from the 'columns' list."
            )
        # allow_non_numeric=True with auto-selected columns: silently
        # drop non-numeric columns.
        selected = [c for c in selected if dtypes[c] in _NUMERIC_DTYPES]
        if not selected:
            raise ValueError(
                "All selected columns are non-numeric and "
                "allow_non_numeric=True filters them out, "
                "leaving no columns to convert."
            )

    cpp_frame = frame._frame
    n_rows = cpp_frame.num_rows()

    # Build a column-name → column-index mapping for O(1) lookup.
    col_index = {name: i for i, name in enumerate(all_columns)}

    arrays: list[np.ndarray] = []
    for col_name in selected:
        col = cpp_frame.column_by_index(col_index[col_name])
        dtype = dtypes[col_name]
        mask = col.get_null_mask()

        if dtype == "int64":
            arr = col.to_numpy_int().astype(np.float64, copy=True)
        elif dtype == "float64":
            arr = col.to_numpy_float().copy()
        elif dtype == "bool":
            arr = col.to_numpy_bool().astype(np.float64, copy=True)
        else:
            # Should not be reached after the dtype filter above.
            raise TypeError(f"Column '{col_name}' has unsupported dtype '{dtype}'.")

        # Apply null mask — replace masked positions with null_value.
        if any(mask):
            null_indices = [i for i, m in enumerate(mask) if m]
            arr[null_indices] = null_value

        arrays.append(arr)

    if n_rows == 0:
        return np.empty((0, len(selected)), dtype=np.float64)

    return np.column_stack(arrays)
