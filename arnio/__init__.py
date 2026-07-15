"""arnio — Data trust for Python.

Validate, clean, and profile DataFrames before everything else.

    import arnio as ar

    result = ar.validate(df, schema)
    report = ar.profile(df)
    cleaned = ar.clean(df, ["strip_whitespace", "drop_duplicates"])
"""

from arnio._version import __version__ as __version__

# -- Functions (primary API surface) ----------------------------------------

from arnio.validate import validate as validate
from arnio.profile import profile as profile
from arnio.profile import suggest as suggest
from arnio.clean import clean as clean
from arnio.gates import check as check
from arnio.schema._inference import infer_schema as infer_schema
from arnio.schema._diff import diff_schemas as diff_schemas

# -- Classes ----------------------------------------------------------------

from arnio.schema import Schema as Schema
from arnio.clean import Pipeline as Pipeline
from arnio.validate import ValidationResult as ValidationResult
from arnio.validate import Issue as Issue
from arnio.profile import ProfileReport as ProfileReport
from arnio.profile import ColumnProfile as ColumnProfile

# -- Field types ------------------------------------------------------------

from arnio.schema import Field as Field
from arnio.schema import Int as Int
from arnio.schema import Float as Float
from arnio.schema import String as String
from arnio.schema import Bool as Bool
from arnio.schema import Date as Date
from arnio.schema import DateTime as DateTime
from arnio.schema import Email as Email
from arnio.schema import URL as URL
from arnio.schema import PhoneNumber as PhoneNumber
from arnio.schema import IPAddress as IPAddress
from arnio.schema import UUID as UUID
from arnio.schema import Regex as Regex

# -- Schema operations ------------------------------------------------------

from arnio.schema._diff import SchemaDiff as SchemaDiff
from arnio.schema._diff import FieldDiff as FieldDiff
from arnio.schema._serde import schema_to_dict as schema_to_dict
from arnio.schema._serde import schema_from_dict as schema_from_dict
from arnio.schema._serde import schema_to_json as schema_to_json
from arnio.schema._serde import schema_from_json as schema_from_json
from arnio.schema._serde import schema_to_yaml as schema_to_yaml
from arnio.schema._serde import schema_from_yaml as schema_from_yaml

# -- Cleaning step registration ---------------------------------------------

from arnio.clean._registry import register_step as register_step
from arnio.clean._registry import unregister_step as unregister_step
from arnio.clean._registry import list_steps as list_steps

# -- Exceptions -------------------------------------------------------------

from arnio.exceptions import ArnioError as ArnioError
from arnio.exceptions import SchemaError as SchemaError
from arnio.exceptions import AdapterError as AdapterError
from arnio.exceptions import ValidationError as ValidationError
from arnio.exceptions import CleaningError as CleaningError
from arnio.exceptions import PipelineError as PipelineError

# -- Integrations (side-effect: registers pandas accessor) ------------------

import arnio.integrations  # noqa: F401

__all__ = [
    # Version
    "__version__",
    # Functions
    "validate",
    "profile",
    "clean",
    "suggest",
    "check",
    "infer_schema",
    "diff_schemas",
    # Classes
    "Schema",
    "Pipeline",
    "ValidationResult",
    "Issue",
    "ProfileReport",
    "ColumnProfile",
    "SchemaDiff",
    "FieldDiff",
    # Field types
    "Field",
    "Int",
    "Float",
    "String",
    "Bool",
    "Date",
    "DateTime",
    "Email",
    "URL",
    "PhoneNumber",
    "IPAddress",
    "UUID",
    "Regex",
    # Schema operations
    "schema_to_dict",
    "schema_from_dict",
    "schema_to_json",
    "schema_from_json",
    "schema_to_yaml",
    "schema_from_yaml",
    # Step registration
    "register_step",
    "unregister_step",
    "list_steps",
    # Exceptions
    "ArnioError",
    "SchemaError",
    "AdapterError",
    "ValidationError",
    "CleaningError",
    "PipelineError",
]
