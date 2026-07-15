"""Tests for the dict adapter."""

import pytest

from arnio.adapt._dict import DictAdapter


@pytest.fixture
def adapter() -> DictAdapter:
    data = [
        {"name": "Alice", "age": 25, "score": 95.5},
        {"name": "Bob", "age": 30, "score": 87.3},
        {"name": "Charlie", "age": 35, "score": 76.1},
    ]
    return DictAdapter(data)


class TestDictAdapter:
    def test_column_names(self, adapter: DictAdapter):
        assert adapter.column_names() == ["name", "age", "score"]

    def test_row_count(self, adapter: DictAdapter):
        assert adapter.row_count() == 3

    def test_null_count(self, adapter: DictAdapter):
        assert adapter.null_count("name") == 0

    def test_unique_count(self, adapter: DictAdapter):
        assert adapter.unique_count("name") == 3

    def test_unwrap_returns_list_of_dicts(self, adapter: DictAdapter):
        result = adapter.unwrap()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert len(result) == 3

    def test_strip_whitespace(self):
        data = [{"name": "  Alice  "}, {"name": "  Bob  "}]
        adapter = DictAdapter(data)
        result = adapter.strip_whitespace()
        unwrapped = result.unwrap()
        assert unwrapped[0]["name"] == "Alice"
        assert unwrapped[1]["name"] == "Bob"

    def test_drop_duplicates(self):
        data = [{"a": 1}, {"a": 1}, {"a": 2}]
        adapter = DictAdapter(data)
        result = adapter.drop_duplicates()
        assert result.row_count() == 2

    def test_column_oriented_dict(self):
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        adapter = DictAdapter(data)
        assert adapter.row_count() == 2
        assert adapter.column_names() == ["name", "age"]

    def test_empty_list(self):
        adapter = DictAdapter([])
        assert adapter.row_count() == 0
        assert adapter.column_names() == []
