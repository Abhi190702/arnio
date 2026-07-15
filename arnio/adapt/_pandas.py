"""pandas DataFrame adapter — implements DataFrameAdapter for pd.DataFrame."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

from arnio.adapt._protocol import DataFrameAdapter, NumericStats, StringLengthStats


def _is_string_dtype(series: pd.Series) -> bool:
    """Check if a pandas Series has string-like dtype (works with pandas 2.x and 3.x)."""
    dtype = series.dtype
    dtype_str = str(dtype).lower()
    return (
        dtype == object
        or dtype.kind in ("U", "S")
        or "string" in dtype_str
        or dtype_str == "str"
    )

# Default tokens treated as missing values by standardize_missing.
_DEFAULT_MISSING_TOKENS: frozenset[str] = frozenset({
    "", "n/a", "N/A", "na", "NA", "nan", "NaN", "NAN",
    "null", "NULL", "none", "None", "NONE",
    "-", "--", ".", "?", "missing", "MISSING",
    "undefined", "UNDEFINED", "#N/A", "#NA", "#REF!",
    "not available", "Not Available",
})


class PandasAdapter:
    """DataFrameAdapter implementation for pandas DataFrames.

    All mutating operations return a NEW PandasAdapter wrapping a copy.
    The original DataFrame is never modified.
    """

    __slots__ = ("_df",)

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # -- Identity & metadata ------------------------------------------------

    def column_names(self) -> list[str]:
        return list(self._df.columns)

    def row_count(self) -> int:
        return len(self._df)

    def column_dtype(self, column: str) -> str:
        dtype = self._df[column].dtype
        kind = dtype.kind

        mapping: dict[str, str] = {
            "i": "int64",
            "u": "int64",
            "f": "float64",
            "b": "bool",
            "M": "datetime",
            "U": "string",
            "S": "string",
        }

        if kind in mapping:
            return mapping[kind]

        # pandas StringDtype, nullable integer, etc.
        dtype_str = str(dtype).lower()
        if "int" in dtype_str:
            return "int64"
        if "float" in dtype_str:
            return "float64"
        if "bool" in dtype_str:
            return "bool"
        if "string" in dtype_str or dtype_str == "str":
            return "string"
        if "datetime" in dtype_str:
            return "datetime"
        if dtype_str == "object":
            return "object"

        return str(dtype)

    # -- Null analysis ------------------------------------------------------

    def null_count(self, column: str) -> int:
        return int(self._df[column].isna().sum())

    # -- Uniqueness ---------------------------------------------------------

    def unique_count(self, column: str) -> int:
        return int(self._df[column].nunique(dropna=True))

    def duplicate_count(self) -> int:
        return int(self._df.duplicated().sum())

    # -- Value inspection ---------------------------------------------------

    def value_counts(self, column: str, *, top_n: int = 10) -> dict[Any, int]:
        counts = self._df[column].value_counts(dropna=True).head(top_n)
        return dict(zip(counts.index.tolist(), counts.values.tolist()))

    def values_in_set(self, column: str, allowed: set[Any]) -> int:
        series = self._df[column].dropna()
        return int(series.isin(allowed).sum())

    def regex_match_count(self, column: str, pattern: str) -> int:
        series = self._df[column].dropna().astype(str)
        return int(series.str.fullmatch(pattern).sum())

    def column_values(self, column: str) -> list[Any]:
        return self._df[column].tolist()

    # -- Numeric statistics -------------------------------------------------

    def numeric_stats(self, column: str) -> NumericStats:
        series = self._df[column].dropna()
        desc = series.describe()
        return NumericStats(
            mean=float(desc.get("mean", 0)),
            std=float(desc.get("std", 0)),
            min=float(desc.get("min", 0)),
            max=float(desc.get("max", 0)),
            median=float(series.median()),
            q1=float(desc.get("25%", 0)),
            q3=float(desc.get("75%", 0)),
        )

    # -- String statistics --------------------------------------------------

    def string_lengths(self, column: str) -> StringLengthStats:
        lengths = self._df[column].dropna().astype(str).str.len()
        if lengths.empty:
            return StringLengthStats(min_length=0, max_length=0, mean_length=0.0)
        return StringLengthStats(
            min_length=int(lengths.min()),
            max_length=int(lengths.max()),
            mean_length=float(lengths.mean()),
        )

    # -- Sampling -----------------------------------------------------------

    def sample(self, n: int) -> PandasAdapter:
        actual_n = min(n, len(self._df))
        if actual_n == 0:
            return PandasAdapter(self._df.head(0))
        return PandasAdapter(self._df.sample(n=actual_n, random_state=42))

    # -- Mutating operations (return new adapter) ---------------------------

    def strip_whitespace(self, columns: list[str] | None = None) -> PandasAdapter:
        df = self._df.copy()
        cols = columns or [c for c in df.columns if _is_string_dtype(df[c])]
        for col in cols:
            if col in df.columns and _is_string_dtype(df[col]):
                df[col] = df[col].str.strip()
        return PandasAdapter(df)

    def normalize_case(
        self, columns: list[str] | None = None, *, case: str = "lower"
    ) -> PandasAdapter:
        df = self._df.copy()
        cols = columns or [c for c in df.columns if _is_string_dtype(df[c])]

        case_fn = {"lower": str.lower, "upper": str.upper, "title": str.title}
        if case not in case_fn:
            raise ValueError(f"case must be 'lower', 'upper', or 'title', got {case!r}")

        for col in cols:
            if col in df.columns and _is_string_dtype(df[col]):
                fn = case_fn[case]
                df[col] = df[col].map(
                    lambda x, _fn=fn: _fn(x) if isinstance(x, str) else x
                )
        return PandasAdapter(df)

    def drop_duplicates(self) -> PandasAdapter:
        return PandasAdapter(self._df.drop_duplicates().reset_index(drop=True))

    def drop_nulls(
        self,
        columns: list[str] | None = None,
        *,
        how: str = "any",
    ) -> PandasAdapter:
        df = self._df.dropna(subset=columns, how=how).reset_index(drop=True)
        return PandasAdapter(df)

    def fill_nulls(self, column: str, value: Any) -> PandasAdapter:
        df = self._df.copy()
        df[column] = df[column].fillna(value)
        return PandasAdapter(df)

    def rename_columns(self, mapping: dict[str, str]) -> PandasAdapter:
        return PandasAdapter(self._df.rename(columns=mapping))

    def drop_columns(self, columns: list[str]) -> PandasAdapter:
        existing = [c for c in columns if c in self._df.columns]
        return PandasAdapter(self._df.drop(columns=existing))

    def cast_column(self, column: str, dtype: str) -> PandasAdapter:
        df = self._df.copy()
        dtype_map: dict[str, str] = {
            "int64": "int64",
            "float64": "float64",
            "string": "object",
            "bool": "bool",
        }
        pd_dtype = dtype_map.get(dtype, dtype)
        df[column] = df[column].astype(pd_dtype)
        return PandasAdapter(df)

    def replace_values(self, column: str, mapping: dict[Any, Any]) -> PandasAdapter:
        df = self._df.copy()
        df[column] = df[column].replace(mapping)
        return PandasAdapter(df)

    def slugify_column_names(self) -> PandasAdapter:
        def _slugify(name: str) -> str:
            # Normalize unicode to ASCII-compatible form
            name = unicodedata.normalize("NFKD", name)
            name = name.encode("ascii", "ignore").decode("ascii")
            # Replace non-alphanumeric with underscores
            name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
            # Strip leading/trailing underscores and collapse doubles
            name = re.sub(r"_+", "_", name).strip("_")
            return name.lower()

        mapping = {col: _slugify(col) for col in self._df.columns}
        return self.rename_columns(mapping)

    def standardize_missing(
        self,
        tokens: set[str] | None = None,
        columns: list[str] | None = None,
    ) -> PandasAdapter:
        df = self._df.copy()
        token_set = tokens if tokens is not None else _DEFAULT_MISSING_TOKENS
        cols = columns or [c for c in df.columns if _is_string_dtype(df[c])]
        for col in cols:
            if col in df.columns and _is_string_dtype(df[col]):
                mask = df[col].isin(token_set)
                df.loc[mask, col] = pd.NA
        return PandasAdapter(df)

    # -- Unwrap -------------------------------------------------------------

    def unwrap(self) -> pd.DataFrame:
        return self._df

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return f"PandasAdapter({self.row_count()} rows, {len(self.column_names())} columns)"
