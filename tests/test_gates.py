"""Tests for quality gates."""

import pandas as pd
import pytest

import arnio as ar
from arnio.exceptions import ValidationError


class TestCheck:
    """Test ar.check() — assertion-style validation."""

    def test_valid_data_passes_silently(self):
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        schema = ar.Schema({"name": ar.String(), "age": ar.Int()})
        # Should not raise
        ar.check(df, schema)

    def test_invalid_data_raises(self):
        df = pd.DataFrame({"name": [None]})
        schema = ar.Schema({"name": ar.String(nullable=False)})
        with pytest.raises(ValidationError) as exc_info:
            ar.check(df, schema)
        assert exc_info.value.issues

    def test_error_message_includes_details(self):
        df = pd.DataFrame({"age": [-1]})
        schema = ar.Schema({"age": ar.Int(min=0)})
        with pytest.raises(ValidationError, match="age"):
            ar.check(df, schema)

    def test_dict_schema_accepted(self):
        df = pd.DataFrame({"x": [1]})
        ar.check(df, {"x": ar.Int()})


class TestPandasAccessor:
    """Test df.arnio.* accessor."""

    def test_validate(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = df.arnio.validate({"x": ar.Int()})
        assert result.passed

    def test_profile(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = df.arnio.profile()
        assert report.row_count == 3

    def test_clean(self):
        df = pd.DataFrame({"name": ["  Alice  "]})
        result = df.arnio.clean(["strip_whitespace"])
        assert result["name"].iloc[0] == "Alice"

    def test_suggest(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        suggestions = df.arnio.suggest()
        assert isinstance(suggestions, list)

    def test_is_valid(self):
        df = pd.DataFrame({"x": [1]})
        assert df.arnio.is_valid({"x": ar.Int()}) is True

    def test_check_passes(self):
        df = pd.DataFrame({"x": [1]})
        df.arnio.check({"x": ar.Int()})

    def test_check_fails(self):
        df = pd.DataFrame({"x": [None]})
        with pytest.raises(ValidationError):
            df.arnio.check({"x": ar.Int(nullable=False)})
