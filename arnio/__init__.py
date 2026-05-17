"""arnio — Fast CSV processing and data cleaning companion for pandas.

import arnio as ar
"""

import ast
import importlib
import os
import sys
from typing import Any

try:
    from importlib.metadata import version

    __version__ = version("arnio")
except Exception:
    __version__ = "unknown"

from .cleaning import (
    cast_types,
    clean,
    clip_numeric,
    drop_constant_columns,
    drop_duplicates,
    drop_nulls,
    fill_nulls,
    filter_rows,
    normalize_case,
    rename_columns,
    round_numeric_columns,
    safe_divide_columns,
    strip_whitespace,
    validate_columns_exist,
)
from .convert import from_pandas, to_numpy, to_pandas
from .exceptions import ArnioError, CsvReadError, TypeCastError, UnknownStepError
from .frame import ArFrame
from .integrations import ArnioPandasAccessor
from .io import read_csv, scan_csv
from .pipeline import pipeline, register_step
from .quality import (
    ColumnProfile,
    DataQualityReport,
    auto_clean,
    profile,
    suggest_cleaning,
)
from .schema import (
    URL,
    Bool,
    Email,
    Field,
    Float64,
    Int64,
    Schema,
    String,
    ValidationIssue,
    ValidationResult,
    validate,
)

__all__ = [
    # Core class
    "ArFrame",
    "ColumnSummary",
    # I/O
    "read_csv",
    "read_csv_chunked",
    "read_jsonl",
    "write_csv",
    "write_parquet",
    "scan_csv",
    "sniff_delimiter",
    # Cleaning
    "drop_nulls",
    "drop_columns",
    "select_columns",
    "keep_rows_with_nulls",
    "fill_nulls",
    "validate_columns_exist",
    "filter_rows",
    "replace_values",
    "drop_duplicates",
    "drop_constant_columns",
    "drop_empty_columns",
    "clip_numeric",
    "winsorize_outliers",
    "coalesce_columns",
    "combine_columns",
    "drop_columns_matching",
    "strip_whitespace",
    "parse_bool_strings",
    "normalize_case",
    "rename_columns",
    "round_numeric_columns",
    "cast_types",
    "clean",
    "winsorize_outliers",
    "safe_divide_columns",
    "trim_column_names",
    "standardize_missing_tokens",
    # Conversion
    "to_pandas",
    "to_numpy",
    "from_pandas",
    "from_records",
    # Integrations
    "ArnioPandasAccessor",
    "register_duckdb",
    # Pipeline
    "pipeline",
    "register_step",
    "get_builtin_step_signatures",
    "list_steps",
    "PipelineContext",
    "reset_steps",
    # Data quality
    "profile",
    "compare_profiles",
    "check_quality_gates",
    "suggest_cleaning",
    "auto_clean",
    "ColumnProfile",
    "DataQualityReport",
    "CleanStepRecord",
    "CleanExplanation",
    "ProfileComparison",
    "QualityGateIssue",
    "QualityGateResult",
    # Schema validation
    "Schema",
    "SchemaDiff",
    "SchemaDiffEntry",
    "Field",
    "ValidationIssue",
    "ValidationResult",
    "validate",
    "diff_schema",
    "Int64",
    "Float64",
    "String",
    "CountryCode",
    "CurrencyCode",
    "Bool",
    "Email",
    "URL",
    "PhoneNumber",
    "DateTime",
    # Exceptions
    "UnknownStepError",
    "ArnioError",
    "CsvReadError",
    "JsonlReadError",
    "TypeCastError",
    "PipelineStepError",
    "normalize_unicode",
    "Regex",
    "Custom",
    "register_validator",
    "Date",
    "schema_to_dict",
    "schema_to_yaml",
]

# 3. Textual AST parsing to pull new developer additions without memory leaks
_CURRENT_DIR = os.path.dirname(__file__)
for submodule_path in _SUBMODULES:
    filename = submodule_path.lstrip(".") + ".py"
    filepath = os.path.join(_CURRENT_DIR, filename)

    if not os.path.exists(filepath):
        continue

    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            if elt.value not in _LAZY_MAPPING:
                                _LAZY_MAPPING[elt.value] = submodule_path
                        elif isinstance(elt, ast.Str):
                            if elt.s not in _LAZY_MAPPING:
                                _LAZY_MAPPING[elt.s] = submodule_path
    except Exception:
        continue

_INTEGRATIONS_LOADED = False


# 4. Use a robust Proxy class module replacement pattern
class _LazyModuleProxy:
    def __init__(self, orig_module: Any):
        self.__dict__["_orig_module"] = orig_module

    def __getattr__(self, name: str) -> Any:
        global _INTEGRATIONS_LOADED

        # Automatically bind pandas integrations ONLY if pandas is already actively running
        if "pandas" in sys.modules and not _INTEGRATIONS_LOADED:
            _INTEGRATIONS_LOADED = True
            try:
                importlib.import_module(
                    ".integrations", self.__dict__["_orig_module"].__name__
                )
            except Exception:
                pass

        if name in _LAZY_MAPPING:
            submodule_name = _LAZY_MAPPING[name]
            submodule = importlib.import_module(
                submodule_name, self.__dict__["_orig_module"].__name__
            )

            if hasattr(submodule, name):
                attr = getattr(submodule, name)
                if isinstance(attr, type(sys)) and attr.__name__.endswith(f".{name}"):
                    if hasattr(attr, name):
                        return getattr(attr, name)
                return attr
            return submodule

        return getattr(self.__dict__["_orig_module"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["_orig_module"], name, value)

    def __dir__(self) -> list[str]:
        return sorted(list(_LAZY_MAPPING.keys()) + ["__version__", "__all__"])


# Fetch the raw original module layout
_orig_module = sys.modules[__name__]
__all__ = list(_LAZY_MAPPING.keys()) + ["__version__"]

# Overwrite system module table entry safely using the proxy boundary shell
sys.modules[__name__] = _LazyModuleProxy(_orig_module)  # type: ignore
