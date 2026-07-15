"""Tests for the profiling engine."""

import pandas as pd
import pytest

import arnio as ar


class TestProfileBasic:
    """Test ar.profile() basic behavior."""

    def test_profile_returns_report(self, sample_df):
        report = ar.profile(sample_df)
        assert isinstance(report, ar.ProfileReport)

    def test_quality_score_range(self, sample_df):
        report = ar.profile(sample_df)
        assert 0 <= report.quality_score <= 100

    def test_row_count(self, sample_df):
        report = ar.profile(sample_df)
        assert report.row_count == 4

    def test_column_count(self, sample_df):
        report = ar.profile(sample_df)
        assert report.column_count == 5

    def test_column_profiles_exist(self, sample_df):
        report = ar.profile(sample_df)
        assert "name" in report.columns
        assert "age" in report.columns

    def test_null_rate_computed(self, sample_df):
        report = ar.profile(sample_df)
        assert report.columns["name"].null_count == 1
        assert report.columns["name"].null_rate == pytest.approx(0.25)

    def test_unique_count_computed(self, sample_df):
        report = ar.profile(sample_df)
        assert report.columns["name"].unique_count == 3

    def test_numeric_stats_for_numeric(self, sample_df):
        report = ar.profile(sample_df)
        assert report.columns["score"].numeric_stats is not None

    def test_string_lengths_for_strings(self, sample_df):
        report = ar.profile(sample_df)
        assert report.columns["name"].string_lengths is not None

    def test_clean_data_high_score(self, clean_df):
        report = ar.profile(clean_df)
        assert report.quality_score >= 80

    def test_duplicate_detection(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        report = ar.profile(df)
        assert report.duplicate_count == 1


class TestProfileOutput:
    """Test ProfileReport output methods."""

    def test_to_dict(self, sample_df):
        report = ar.profile(sample_df)
        d = report.to_dict()
        assert "quality_score" in d
        assert "columns" in d

    def test_to_json(self, sample_df):
        report = ar.profile(sample_df)
        json_str = report.to_json()
        assert "quality_score" in json_str

    def test_to_markdown(self, sample_df):
        report = ar.profile(sample_df)
        md = report.to_markdown()
        assert "Quality Score" in md

    def test_to_html(self, sample_df):
        report = ar.profile(sample_df)
        html = report.to_html()
        assert "arnio-profile" in html

    def test_repr(self, sample_df):
        report = ar.profile(sample_df)
        assert "ProfileReport" in repr(report)


class TestProfileWarnings:
    """Test quality warning detection."""

    def test_all_null_warning(self):
        df = pd.DataFrame({"a": [None, None, None]})
        report = ar.profile(df)
        assert "all_null" in report.columns["a"].warnings

    def test_constant_warning(self):
        df = pd.DataFrame({"a": [42, 42, 42]})
        report = ar.profile(df)
        assert "constant" in report.columns["a"].warnings


class TestProfileWithDicts:
    """Test profiling with dict input."""

    def test_list_of_dicts(self, sample_dicts):
        report = ar.profile(sample_dicts)
        assert report.row_count == 3


class TestSuggest:
    """Test ar.suggest()."""

    def test_suggest_returns_list(self, messy_df):
        suggestions = ar.suggest(messy_df)
        assert isinstance(suggestions, list)

    def test_duplicate_suggestion(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        suggestions = ar.suggest(df)
        step_names = [s["step"] for s in suggestions]
        assert "drop_duplicates" in step_names

    def test_all_null_suggests_drop(self):
        df = pd.DataFrame({"good": [1, 2, 3], "bad": [None, None, None]})
        suggestions = ar.suggest(df)
        step_names = [s["step"] for s in suggestions]
        assert "drop_columns" in step_names

    def test_suggestion_has_required_keys(self, messy_df):
        suggestions = ar.suggest(messy_df)
        if suggestions:
            s = suggestions[0]
            assert "step" in s
            assert "reason" in s
            assert "confidence" in s
