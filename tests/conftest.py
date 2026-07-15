"""Shared fixtures for arnio tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small DataFrame with mixed types for testing."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", None],
        "age": [25, 30, 35, None],
        "email": ["alice@example.com", "bob@example.com", "invalid", None],
        "score": [95.5, 87.3, 76.1, None],
        "active": [True, False, True, None],
    })


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A clean DataFrame with no quality issues."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
    })


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """A messy DataFrame for testing cleaning operations."""
    return pd.DataFrame({
        "Name ": ["  Alice  ", "bob", "CHARLIE", "  Alice  "],
        "Age": [25, None, 35, 25],
        "Email Address": ["alice@test.com", "N/A", "bad-email", "alice@test.com"],
    })


@pytest.fixture
def sample_dicts() -> list[dict]:
    """A list of dicts for testing the dict adapter."""
    return [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
        {"name": "Charlie", "age": 35},
    ]
