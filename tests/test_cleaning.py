"""Tests for data cleaning functions."""

import pandas as pd
import pytest

import arnio as ar
from arnio import from_pandas, to_pandas
from arnio.cleaning import _validate_column_sequence, _validate_string_mapping


class TestDropNulls:
    def test_drop_all_nulls(self, csv_with_nulls):
        frame = ar.read_csv(csv_with_nulls)
        result = ar.drop_nulls(frame)
        assert result.shape[0] < frame.shape[0]
        # Only Alice and Diana have no nulls
        assert result.shape[0] == 2

    def test_drop_nulls_subset(self, csv_with_nulls):
        frame = ar.read_csv(csv_with_nulls)
        result = ar.drop_nulls(frame, subset=["name"])
        # Only row 2 has null name
        assert result.shape[0] == 3

    def test_drop_nulls_empty_subset_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None, "Charlie"]}))

        with pytest.raises(ValueError, match="subset"):
            ar.drop_nulls(frame, subset=[])

    def test_drop_nulls_pipeline_empty_subset_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None, "Charlie"]}))

        with pytest.raises(ValueError, match="subset"):
            ar.pipeline(frame, [("drop_nulls", {"subset": []})])

    def test_drop_nulls_subset_none_still_works(self):
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None, "Charlie"]}))

        result = ar.drop_nulls(frame)

        assert result.shape[0] == 2


class TestKeepRowsWithNulls:
    def test_pandas_input_empty_subset_raises(self):
        df = pd.DataFrame({"name": ["Alice", None]})

        with pytest.raises(ValueError, match="subset"):
            ar.keep_rows_with_nulls(df, subset=[])

    def test_pandas_input_empty_tuple_subset_raises(self):
        df = pd.DataFrame({"name": ["Alice", None]})

        with pytest.raises(ValueError, match="subset"):
            ar.keep_rows_with_nulls(df, subset=())

    def test_empty_subset_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None]}))

        with pytest.raises(ValueError, match="subset"):
            ar.keep_rows_with_nulls(frame, subset=[])

    def test_empty_tuple_subset_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None]}))

        with pytest.raises(ValueError, match="subset"):
            ar.keep_rows_with_nulls(frame, subset=())

    def test_subset_none_still_works(self):
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None]}))

        result = ar.keep_rows_with_nulls(frame)

        assert result.shape[0] == 1

    def test_keeps_only_null_rows(self, csv_with_nulls):
        # full frame has 4 rows, 2 have nulls (row1: null name+score, row2: null age)
        frame = ar.read_csv(csv_with_nulls)
        result = ar.keep_rows_with_nulls(frame)
        assert result.shape[0] == 2

    def test_no_nulls_returns_empty(self, sample_csv):
        # sample_csv has no nulls — result should be empty
        frame = ar.read_csv(sample_csv)
        result = ar.keep_rows_with_nulls(frame)
        assert result.shape[0] == 0

    def test_all_nulls_returns_all_rows(self, tmp_path):
        # every row has a null — all rows should be kept
        path = tmp_path / "all_nulls.csv"
        path.write_text("name,age\nAlice,\n,25\nCharlie,\n")
        frame = ar.read_csv(path)
        result = ar.keep_rows_with_nulls(frame)
        assert result.shape[0] == frame.shape[0]

    def test_subset_targets_specific_column(self, csv_with_nulls):
        # only checking 'age' column — only Charlie has null age
        frame = ar.read_csv(csv_with_nulls)
        result = ar.keep_rows_with_nulls(frame, subset=["age"])
        assert result.shape[0] == 1

    def test_subset_unknown_column_raises(self, csv_with_nulls):
        # passing a column that doesn't exist should raise ValueError
        frame = ar.read_csv(csv_with_nulls)
        with pytest.raises(KeyError):
            ar.keep_rows_with_nulls(frame, subset=["nonexistent"])

    def test_index_is_reset(self, csv_with_nulls):
        # returned frame should have clean 0-based index
        frame = ar.read_csv(csv_with_nulls)
        result = ar.keep_rows_with_nulls(frame)
        df = ar.to_pandas(result)
        assert list(df.index) == list(range(len(df)))

    def test_pipeline_usage(self, csv_with_nulls):
        # function should work correctly when called via pipeline
        frame = ar.read_csv(csv_with_nulls)
        result = ar.pipeline(
            frame,
            [
                ("keep_rows_with_nulls",),
            ],
        )
        assert result.shape[0] == 2

    def test_pipeline_subset(self, csv_with_nulls):
        # pipeline with subset parameter
        frame = ar.read_csv(csv_with_nulls)
        result = ar.pipeline(
            frame,
            [
                ("keep_rows_with_nulls", {"subset": ["age"]}),
            ],
        )
        assert result.shape[0] == 1

    def test_invalid_subset_string(self, csv_with_nulls):
        """keep_rows_with_nulls raises TypeError when subset is a string."""
        frame = ar.read_csv(csv_with_nulls)
        with pytest.raises(TypeError, match="must be a list"):
            ar.keep_rows_with_nulls(frame, subset="age")

    def test_missing_column_raises(self, csv_with_nulls):
        """keep_rows_with_nulls raises KeyError when subset column is missing."""
        frame = ar.read_csv(csv_with_nulls)
        with pytest.raises(KeyError, match="nonexistent"):
            ar.keep_rows_with_nulls(frame, subset=["nonexistent"])


class TestFillNulls:
    def test_fill_with_string(self, csv_with_nulls):
        frame = ar.read_csv(csv_with_nulls)
        result = ar.fill_nulls(frame, "N/A", subset=["name"])
        assert result.shape == frame.shape

    def test_fill_with_number(self, csv_with_nulls):
        frame = ar.read_csv(csv_with_nulls)
        result = ar.fill_nulls(frame, 0)
        assert result.shape == frame.shape

    def test_incompatible_fill_rejected(self, tmp_path):
        path = tmp_path / "numbers.csv"
        path.write_text("x,y\n1,a\n,b\n3,c\n")
        frame = ar.read_csv(path)

        with pytest.raises(ValueError, match="Fill value is incompatible"):
            ar.fill_nulls(frame, "bad", subset=["x"])

    def test_fill_nulls_rejects_unsupported_types(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None], "b": ["x", None]}))

        for bad_value in [[1, 2], {"key": "val"}, object()]:
            with pytest.raises(
                TypeError, match="fill value must be a supported scalar"
            ):
                ar.fill_nulls(frame, bad_value)

    def test_fill_nulls_accepts_valid_scalars(self):
        # numeric column → fill with numeric
        frame_num = ar.from_pandas(pd.DataFrame({"a": [1.0, None]}))
        for good_value in [0, 0.0]:
            result = ar.fill_nulls(frame_num, good_value)
            df = ar.to_pandas(result)
            assert (
                df["a"].isnull().sum() == 0
            ), f"Nulls remain after filling with {good_value!r}"

        # string column → fill with string
        frame_str = ar.from_pandas(pd.DataFrame({"b": ["x", None]}))
        result = ar.fill_nulls(frame_str, "missing")
        df = ar.to_pandas(result)
        assert df["b"].isnull().sum() == 0, "Nulls remain after filling with 'missing'"

    def test_fill_nulls_rejects_bool_for_int64_column(self):
        frame = ar.from_pandas(
            pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64")})
        )
        with pytest.raises(TypeError, match="bool"):
            ar.fill_nulls(frame, True)

    def test_fill_nulls_rejects_bool_for_float64_column(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1.0, None, 3.0]}))
        with pytest.raises(TypeError, match="bool"):
            ar.fill_nulls(frame, False)

    def test_fill_nulls_rejects_bool_via_subset(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {"a": pd.array([1, None, 3], dtype="Int64"), "b": ["x", None, "z"]}
            )
        )
        with pytest.raises(TypeError, match="bool"):
            ar.fill_nulls(frame, True, subset=["a"])

    def test_fill_nulls_bool_accepted_for_bool_column(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [True, None, False]}))
        result = ar.fill_nulls(frame, False)
        assert result is not None


class TestWinsorizeOutliers:
    def test_winsorize_outliers_clips_numeric_values(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "value": [1, 2, 3, 4, 100],
                }
            )
        )

        result = ar.winsorize_outliers(frame, lower=0.2, upper=0.8)
        df = ar.to_pandas(result)

        assert df["value"].tolist() == pytest.approx([1.8, 2.0, 3.0, 4.0, 23.2])

    def test_winsorize_outliers_subset(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "a": [1, 2, 3, 4, 100],
                    "b": [10, 20, 30, 40, 500],
                }
            )
        )

        result = ar.winsorize_outliers(
            frame,
            lower=0.2,
            upper=0.8,
            subset=["a"],
        )
        df = ar.to_pandas(result)

        assert df["a"].tolist() == pytest.approx([1.8, 2.0, 3.0, 4.0, 23.2])
        assert list(df["b"]) == [10, 20, 30, 40, 500]

    def test_winsorize_outliers_ignores_non_numeric_without_subset(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "value": [1, 2, 3, 4, 100],
                    "label": ["a", "b", "c", "d", "e"],
                }
            )
        )

        result = ar.winsorize_outliers(frame, lower=0.2, upper=0.8)
        df = ar.to_pandas(result)

        assert df["value"].tolist() == pytest.approx([1.8, 2.0, 3.0, 4.0, 23.2])
        assert list(df["label"]) == ["a", "b", "c", "d", "e"]

    def test_winsorize_outliers_rejects_non_numeric_subset(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "value": [1, 2, 3],
                    "label": ["a", "b", "c"],
                }
            )
        )

        with pytest.raises(ValueError, match="only supports numeric columns"):
            ar.winsorize_outliers(frame, subset=["label"])

    def test_winsorize_outliers_rejects_unknown_subset_column(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.winsorize_outliers(frame, subset=["missing"])

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            (-0.1, 0.95),
            (0.05, 1.1),
            (0.8, 0.2),
            (0.5, 0.5),
        ],
    )
    def test_winsorize_outliers_rejects_invalid_bounds(self, lower, upper):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(ValueError):
            ar.winsorize_outliers(frame, lower=lower, upper=upper)

    def test_winsorize_outliers_rejects_boolean_bounds(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(TypeError, match="'lower' must be an int or float"):
            ar.winsorize_outliers(frame, lower=True)

        with pytest.raises(TypeError, match="'upper' must be an int or float"):
            ar.winsorize_outliers(frame, upper=False)

    @pytest.mark.parametrize("value", ["0.1", None, object()])
    def test_winsorize_outliers_rejects_non_numeric_bounds(self, value):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(TypeError, match="'lower' must be an int or float"):
            ar.winsorize_outliers(frame, lower=value)

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_winsorize_outliers_rejects_non_finite_bounds(self, value):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(ValueError, match="finite"):
            ar.winsorize_outliers(frame, lower=value)

    def test_winsorize_outliers_identical_values_noop(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [5, 5, 5]}))

        result = ar.winsorize_outliers(frame)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [5, 5, 5]

    def test_winsorize_outliers_single_row_noop(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [10]}))

        result = ar.winsorize_outliers(frame)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [10]

    def test_winsorize_outliers_empty_dataframe(self):
        frame = ar.from_pandas(pd.DataFrame({"value": pd.Series(dtype="float64")}))
        result = ar.winsorize_outliers(frame)
        df = ar.to_pandas(result)
        assert df.shape == (0, 1)

    def test_winsorize_outliers_all_nulls(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [None, None, None]}))
        result = ar.winsorize_outliers(frame)
        df = ar.to_pandas(result)
        assert df["value"].isna().all()

    def test_winsorize_outliers_two_rows(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [10.0, 20.0]}))
        result = ar.winsorize_outliers(frame, lower=0.1, upper=0.9)
        df = ar.to_pandas(result)
        assert df["value"].tolist() == pytest.approx([11.0, 19.0])


class TestValidateColumnsExist:
    def test_returns_original_frame_when_columns_exist(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.validate_columns_exist(frame, ["name", "age"])

        assert result is frame

    def test_allows_empty_column_list(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.validate_columns_exist(frame, [])

        assert result is frame

    def test_raises_clear_error_for_missing_columns(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(KeyError, match="Missing columns for test_op"):
            ar.validate_columns_exist(frame, ["missing"], operation="test_op")

    def test_rejects_string_columns_argument(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="not a string"):
            ar.validate_columns_exist(frame, "name")

    def test_rejects_non_string_column_items(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="only string column names"):
            ar.validate_columns_exist(frame, ["name", 1])

    def test_drop_nulls_rejects_string_subset(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="subset must be a sequence"):
            ar.drop_nulls(frame, subset="name")

    def test_drop_nulls_rejects_missing_subset_column(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(KeyError, match="Missing columns for drop_nulls"):
            ar.drop_nulls(frame, subset=["missing"])

    def test_rename_rejects_missing_mapping_column(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(KeyError, match="Missing columns for rename_columns"):
            ar.rename_columns(frame, {"missing": "new_name"})


class TestSharedColumnSequenceValidation:
    @pytest.mark.parametrize(
        ("func", "kwargs", "error_type", "message"),
        [
            (
                "keep_rows_with_nulls",
                {"subset": ["missing"]},
                KeyError,
                "Missing columns for keep_rows_with_nulls",
            ),
            (
                "fill_nulls",
                {"value": 0, "subset": ["missing"]},
                KeyError,
                "Missing columns for fill_nulls",
            ),
            (
                "drop_duplicates",
                {"subset": ["missing"]},
                KeyError,
                "Missing columns for drop_duplicates",
            ),
            (
                "strip_whitespace",
                {"subset": ["missing"]},
                KeyError,
                "Missing columns for strip_whitespace",
            ),
            (
                "normalize_case",
                {"subset": ["missing"]},
                KeyError,
                "Missing columns for normalize_case",
            ),
            (
                "normalize_unicode",
                {"subset": ["missing"]},
                KeyError,
                "Missing columns for normalize_unicode",
            ),
            (
                "standardize_missing_tokens",
                {"subset": ["missing"]},
                ValueError,
                "Unknown columns in subset",
            ),
            (
                "coalesce_columns",
                {"subset": ["missing"]},
                KeyError,
                "Missing columns for coalesce_columns",
            ),
        ],
    )
    def test_shared_subset_validation_rejects_missing_columns(
        self,
        sample_csv,
        func,
        kwargs,
        error_type,
        message,
    ):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(error_type, match=message):
            getattr(ar, func)(frame, **kwargs)

    def test_coalesce_columns_selects_first_non_null_value(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "nickname": [None, "Bee", None],
                    "name": ["Alice", "Bob", "Cara"],
                }
            )
        )

        result = ar.coalesce_columns(
            frame,
            subset=["nickname", "name"],
            output_column="display_name",
        )
        df = ar.to_pandas(result)

        assert df["display_name"].tolist() == ["Alice", "Bee", "Cara"]

    def test_coalesce_columns_rejects_empty_subset(self):
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice"]}))

        with pytest.raises(ValueError, match="subset must contain at least one column"):
            ar.coalesce_columns(frame, subset=[])

    def test_coalesce_columns_allows_tuple(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "nickname": [None, "Bee", None],
                    "name": ["Alice", "Bob", "Cara"],
                }
            )
        )

        result = ar.coalesce_columns(
            frame,
            subset=("nickname", "name"),
            output_column="display_name",
        )
        df = ar.to_pandas(result)

        assert df["display_name"].tolist() == ["Alice", "Bee", "Cara"]

    @pytest.mark.parametrize(
        ("func", "kwargs", "message"),
        [
            ("drop_columns", {"columns": 123}, "must be a sequence of column names"),
            (
                "fill_nulls",
                {"value": 0, "subset": 123},
                "must be a sequence of column names",
            ),
            ("drop_duplicates", {"subset": 123}, "must be a sequence of column names"),
            (
                "strip_whitespace",
                {"subset": 123},
                "must be a sequence of column names",
            ),
            ("normalize_case", {"subset": 123}, "must be a sequence of column names"),
            (
                "normalize_unicode",
                {"subset": 123},
                "must be a sequence of column names",
            ),
            (
                "combine_columns",
                {"subset": 123, "separator": "-", "output_column": "combined"},
                "must be a sequence of column names",
            ),
            (
                "coalesce_columns",
                {"subset": 123, "output_column": "combined"},
                "must be a sequence of column names",
            ),
        ],
    )
    def test_shared_subset_validation_rejects_non_sequence_types(
        self,
        sample_csv,
        func,
        kwargs,
        message,
    ):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match=message):
            getattr(ar, func)(frame, **kwargs)

    def test_drop_columns_allows_duplicate_entries(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "debug": ["x", "y"],
                    "name": ["Alice", "Bob"],
                }
            )
        )

        result = ar.drop_columns(frame, ["debug", "debug"])
        df = ar.to_pandas(result)

        assert list(df.columns) == ["id", "name"]

    def test_combine_columns_rejects_duplicate_subset_entries(self):
        frame = ar.from_pandas(pd.DataFrame({"word": ["go"], "suffix": ["!"]}))

        with pytest.raises(ValueError, match="duplicate column names"):
            ar.combine_columns(
                frame,
                subset=["word", "word", "suffix"],
                separator="-",
                output_column="combined",
            )

    def test_combine_columns_rejects_duplicate_subset_direct(self):
        frame = ar.from_pandas(pd.DataFrame({"a": ["x"], "b": ["y"]}))

        with pytest.raises(ValueError, match="duplicate column names"):
            ar.combine_columns(frame, subset=["a", "a"], output_column="combined")

    def test_combine_columns_pipeline_rejects_duplicate_subset(self):
        frame = ar.from_pandas(pd.DataFrame({"a": ["x"], "b": ["y"]}))

        with pytest.raises(ValueError, match="duplicate column names"):
            ar.pipeline(
                frame,
                [
                    (
                        "combine_columns",
                        {"subset": ["a", "a"], "output_column": "combined"},
                    )
                ],
            )

    def test_coalesce_columns_rejects_duplicate_subset_entries(self):
        frame = ar.from_pandas(
            pd.DataFrame({"nickname": [None, "Bee"], "name": ["Alice", "Bob"]})
        )

        with pytest.raises(ValueError, match="duplicate column names"):
            ar.coalesce_columns(
                frame,
                subset=["nickname", "nickname"],
                output_column="display_name",
            )

    def test_coalesce_columns_pipeline_rejects_duplicate_subset(self):
        frame = ar.from_pandas(
            pd.DataFrame({"nickname": [None, "Bee"], "name": ["Alice", "Bob"]})
        )

        with pytest.raises(ValueError, match="duplicate column names"):
            ar.pipeline(
                frame,
                [
                    (
                        "coalesce_columns",
                        {
                            "subset": ["nickname", "nickname"],
                            "output_column": "display_name",
                        },
                    )
                ],
            )


class TestValidateColumnsExist:
    def test_returns_original_frame_when_columns_exist(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.validate_columns_exist(frame, ["name", "age"])

        assert result is frame

    def test_allows_empty_column_list(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.validate_columns_exist(frame, [])

        assert result is frame

    def test_raises_clear_error_for_missing_columns(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(KeyError, match="Missing columns for test_op"):
            ar.validate_columns_exist(frame, ["missing"], operation="test_op")

    def test_rejects_string_columns_argument(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="not a string"):
            ar.validate_columns_exist(frame, "name")

    def test_rejects_non_string_column_items(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="only string column names"):
            ar.validate_columns_exist(frame, ["name", 1])

    def test_drop_nulls_rejects_string_subset(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="subset must be a sequence"):
            ar.drop_nulls(frame, subset="name")

    def test_drop_nulls_rejects_missing_subset_column(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(KeyError, match="Missing columns for drop_nulls"):
            ar.drop_nulls(frame, subset=["missing"])

    def test_rename_rejects_missing_mapping_column(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(KeyError, match="Missing columns for rename_columns"):
            ar.rename_columns(frame, {"missing": "new_name"})


class TestDropDuplicates:
    def test_drop_dupes_first(self, csv_with_duplicates):
        frame = ar.read_csv(csv_with_duplicates)
        result = ar.drop_duplicates(frame)
        assert result.shape[0] == 3  # Alice, Bob, Charlie

    def test_drop_dupes_last(self, csv_with_duplicates):
        frame = ar.read_csv(csv_with_duplicates)
        result = ar.drop_duplicates(frame, keep="last")
        assert result.shape[0] == 3

    def test_drop_dupes_none(self, csv_with_duplicates):
        frame = ar.read_csv(csv_with_duplicates)
        result = ar.drop_duplicates(frame, keep="none")
        # Only Charlie is unique
        assert result.shape[0] == 1

    def test_drop_dupes_false_alias(self, csv_with_duplicates):
        frame = ar.read_csv(csv_with_duplicates)
        result = ar.drop_duplicates(frame, keep=False)
        # Only Charlie is unique
        assert result.shape[0] == 1

    @pytest.mark.parametrize(
        "keep",
        ["invalid", "FIRST", "all", "", True, None],
    )
    def test_drop_dupes_rejects_invalid_keep_values(self, csv_with_duplicates, keep):
        frame = ar.read_csv(csv_with_duplicates)
        with pytest.raises(ValueError, match="keep must be one of"):
            ar.drop_duplicates(frame, keep=keep)

    def test_drop_dupes_subset(self, csv_with_duplicates):
        frame = ar.read_csv(csv_with_duplicates)
        result = ar.drop_duplicates(frame, subset=["name"])
        assert result.shape[0] == 3

    def test_drop_duplicates_empty_subset_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}))

        with pytest.raises(ValueError, match="subset"):
            ar.drop_duplicates(frame, subset=[])

    def test_drop_duplicates_pipeline_empty_subset_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}))

        with pytest.raises(ValueError, match="subset"):
            ar.pipeline(frame, [("drop_duplicates", {"subset": []})])

    def test_drop_duplicates_valid_subset_still_works(self):
        frame = ar.from_pandas(
            pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Alice", "Bob"]})
        )

        result = ar.drop_duplicates(frame, subset=["name"])
        df = ar.to_pandas(result)

        assert result.shape[0] < frame.shape[0]
        assert "name" in df.columns

    def test_drop_duplicates_subset_none_still_works(self):
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 1, 2], "name": ["a", "a", "b"]}))

        result = ar.drop_duplicates(frame)

        assert result.shape[0] == 2

    def test_drop_dupes_regression_keep_true(self, csv_with_duplicates):
        frame = ar.read_csv(csv_with_duplicates)

        with pytest.raises(ValueError, match="keep must be one of"):
            ar.drop_duplicates(frame, keep=True)

    @pytest.mark.parametrize(
        ("keep", "expected_names"),
        [
            ("first", ["Alice", "Bob", "Charlie"]),
            ("last", ["Alice", "Charlie", "Bob"]),
            ("none", ["Charlie"]),
            (False, ["Charlie"]),
        ],
    )
    def test_drop_duplicates_keep_matrix_deterministic(
        self,
        csv_with_duplicates,
        keep,
        expected_names,
    ):
        frame = ar.read_csv(csv_with_duplicates)

        result = ar.drop_duplicates(frame, keep=keep)

        names = ar.to_pandas(result)["name"].tolist()

        assert names == expected_names

    def test_drop_duplicates_type_collision_int_vs_string(self):
        """int 1 and string '1' must NOT be treated as duplicates (fixes #33)."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2], "val": [1, "1"]}))
        result = ar.drop_duplicates(frame)
        assert result.shape[0] == 2

    def test_drop_duplicates_null_vs_empty_string(self):
        """None and '' must NOT be treated as duplicates (fixes #33)."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [None, "", None]}))
        result = ar.drop_duplicates(frame)
        assert result.shape[0] == 2

    def test_drop_duplicates_separator_injection_unit_sep(self):
        """Rows whose values shift around the \x1f boundary must stay distinct (fixes #33).

        With the old row_key (no length prefixing):
          row 0: col1='a'      col2='b\x1fc'  -> key 'a\x1fb\x1fc\x1f'  (BUG: same as row 1)
          row 1: col1='a\x1fb' col2='c'       -> key 'a\x1fb\x1fc\x1f'  (BUG: same as row 0)
        The two rows are distinct but were incorrectly treated as duplicates.
        """
        frame = ar.from_pandas(
            pd.DataFrame({"col1": ["a", "a\x1fb"], "col2": ["b\x1fc", "c"]})
        )
        result = ar.drop_duplicates(frame)
        assert result.shape[0] == 2

    def test_drop_duplicates_separator_injection_colon(self):
        """Values containing ':' must not produce false duplicates (fixes #33)."""
        frame = ar.from_pandas(
            pd.DataFrame({"col1": ["a:b", "a"], "col2": ["c", "b:c"]})
        )
        result = ar.drop_duplicates(frame)
        assert result.shape[0] == 2

    def test_drop_duplicates_separator_injection_synthetic_prefix(self):
        """Values that look like serialized prefixes must not collide (fixes #33)."""
        frame = ar.from_pandas(
            pd.DataFrame({"col1": ["S1:a", ""], "col2": ["b", "S1:ab"]})
        )
        result = ar.drop_duplicates(frame)
        assert result.shape[0] == 2

    def test_drop_dupes_with_nan_and_nulls(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "val1": [np.nan, np.nan, 10.5, 10.5, None, None],
                "val2": ["a", "a", "b", "b", None, None],
            }
        )
        frame = ar.from_pandas(df)

        # keep="first"
        res_first = ar.to_pandas(ar.drop_duplicates(frame, keep="first"))
        assert len(res_first) == 3

        # keep="none"
        res_none = ar.to_pandas(ar.drop_duplicates(frame, keep="none"))
        assert len(res_none) == 0

        # subset with NaN
        res_subset = ar.to_pandas(ar.drop_duplicates(frame, subset=["val1"]))
        assert len(res_subset) == 2

    def test_drop_duplicates_zero_col_subset_none_preserves_rows(self):
        frame = ar.from_pandas(pd.DataFrame(index=range(3)))
        result = ar.drop_duplicates(frame)
        assert result.shape == (3, 0)

    def test_drop_duplicates_zero_col_empty_subset_raises(self):
        frame = ar.from_pandas(pd.DataFrame(index=range(3)))
        with pytest.raises(ValueError, match="subset cannot be empty"):
            ar.drop_duplicates(frame, subset=[])

    def test_drop_duplicates_zero_col_missing_column_raises(self):
        frame = ar.from_pandas(pd.DataFrame(index=range(3)))
        with pytest.raises(KeyError):
            ar.drop_duplicates(frame, subset=["missing"])


class TestDropColumns:
    def test_drop_columns_removes_requested_columns_and_preserves_order(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "debug": ["x", "y"],
                    "name": ["Alice", "Bob"],
                    "flag": [True, False],
                }
            )
        )

        result = ar.drop_columns(frame, ["debug", "flag"])
        df = ar.to_pandas(result)

        assert list(df.columns) == ["id", "name"]
        assert list(df["name"]) == ["Alice", "Bob"]

    def test_drop_columns_accepts_tuple_input(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "a": [1],
                    "b": [2],
                    "c": [3],
                }
            )
        )

        result = ar.drop_columns(frame, ("a",))
        df = ar.to_pandas(result)

        assert list(df.columns) == ["b", "c"]

    def test_drop_columns_allows_empty_input_as_no_op(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result_helper = ar.drop_columns(frame, [])
        result_method = frame.drop_columns([])

        assert result_helper is not frame
        assert result_helper == frame
        assert result_helper == result_method

    def test_drop_columns_rejects_missing_columns(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(ValueError, match="Columns not found in frame"):
            ar.drop_columns(frame, ["missing"])

    def test_drop_columns_rejects_string_input(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="not a string"):
            ar.drop_columns(frame, "age")

    def test_drop_columns_rejects_non_string_items(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="only string column names"):
            ar.drop_columns(frame, ["age", 1])

    def test_drop_columns_rejects_removing_all_columns_across_entry_points(self):
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}))

        with pytest.raises(ValueError, match="drop_columns cannot remove all columns"):
            frame.drop_columns(["id", "name"])

        with pytest.raises(ValueError, match="drop_columns cannot remove all columns"):
            ar.drop_columns(frame, ["id", "name"])

        with pytest.raises(ValueError, match="drop_columns cannot remove all columns"):
            ar.pipeline(
                frame,
                [
                    ("drop_columns", {"columns": ["id", "name"]}),
                ],
            )


class TestDropEmptyColumnsPipeline:
    def test_drop_empty_columns_all_empty(self, csv_with_empty_columns):
        frame = ar.read_csv(csv_with_empty_columns)
        result = ar.drop_empty_columns(frame)
        assert "empty_num" not in result.columns
        assert "empty_text" not in result.columns
        assert "name" in result.columns
        assert "age" in result.columns

    def test_drop_empty_columns_no_empty(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        result = ar.drop_empty_columns(frame)
        assert result.columns == frame.columns
        assert result.shape == frame.shape

    def test_drop_empty_columns_partially_empty(self, tmp_path):
        path = tmp_path / "mixed.csv"
        path.write_text("id,value,mixed\n1,10,\n2,20,data\n3,30,\n")
        frame = ar.read_csv(path)
        result = ar.drop_empty_columns(frame)
        assert "mixed" in result.columns

    def test_drop_empty_columns_pipeline(self, csv_with_empty_columns):
        frame = ar.read_csv(csv_with_empty_columns)
        result = ar.pipeline(
            frame,
            [("drop_empty_columns",)],
        )
        assert "empty_num" not in result.columns
        assert "empty_text" not in result.columns

    def test_drop_empty_columns_empty_frame(self):
        frame = ar.from_pandas(pd.DataFrame(columns=["a", "b", "c"]))
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["a", "b", "c"]
        assert result.shape == frame.shape


class TestDropConstantColumns:
    def test_drop_constant_columns_removes_constant_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "value": [1, 2, 3],
                    "constant_num": [7, 7, 7],
                    "constant_text": ["x", "x", "x"],
                }
            )
        )

        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)

        assert list(df.columns) == ["value"]
        assert list(df["value"]) == [1, 2, 3]

    def test_drop_constant_columns_keeps_non_constant_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "a": [1, 2, 1],
                    "b": ["x", "y", "x"],
                }
            )
        )

        result = ar.drop_constant_columns(frame)

        assert result.columns == frame.columns
        assert result.shape == frame.shape

    def test_drop_constant_columns_drops_all_null_column(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "all_null": [None, None],
                    "value": [1, 2],
                }
            )
        )

        result = ar.drop_constant_columns(frame)

        assert result.columns == ["value"]

    def test_drop_constant_columns_keeps_value_plus_null_column(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "maybe_constant": [1, 1, None],
                    "constant": [2, 2, 2],
                }
            )
        )

        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)

        assert list(df.columns) == ["maybe_constant"]
        assert df.shape == (3, 1)

    def test_drop_constant_columns_empty_frame_keeps_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "empty_num": pd.Series(dtype="float64"),
                    "empty_text": pd.Series(dtype="object"),
                }
            )
        )

        result = ar.drop_constant_columns(frame)

        assert result.columns == ["empty_num", "empty_text"]
        assert result.shape == frame.shape

    def test_drop_constant_columns_all_columns_dropped_preserves_row_count(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": ["x"], "c": [None]}))

        result = ar.drop_constant_columns(frame)

        assert result.columns == []
        assert result.shape[0] == 1
        assert result.shape[1] == 0
        assert ar.to_pandas(result).shape == (1, 0)

    def test_drop_constant_columns_all_columns_dropped_preserves_row_count_multiple_rows(
        self,
    ):
        frame = ar.from_pandas(pd.DataFrame({"a": [7, 7, 7], "b": ["x", "x", "x"]}))
        result = ar.drop_constant_columns(frame)
        assert result.columns == []
        assert result.shape == (3, 0)
        assert ar.to_pandas(result).shape == (3, 0)

    def test_zero_column_frame_shape_and_num_rows(self):
        df = pd.DataFrame(index=range(5))
        frame = ar.from_pandas(df)
        assert frame.shape == (5, 0)
        assert frame.shape[0] == 5
        assert frame.shape[1] == 0

    def test_zero_column_frame_pandas_roundtrip(self):
        for n in [0, 1, 5, 100]:
            df = pd.DataFrame(index=range(n))
            frame = ar.from_pandas(df)
            result = ar.to_pandas(frame)
            assert result.shape == (n, 0), f"failed for n={n}"

    def test_zero_column_frame_clone_preserves_row_count(self):
        df = pd.DataFrame(index=range(4))
        frame = ar.from_pandas(df)
        cloned = frame._frame.clone()
        assert cloned.num_rows() == 4
        assert cloned.num_cols() == 0

    def test_drop_constant_columns_pandas_input(self):
        df = pd.DataFrame(
            {
                "value": [1, 2, 3],
                "constant_num": [7, 7, 7],
                "constant_text": ["x", "x", "x"],
            }
        )

        result = ar.drop_constant_columns(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["value"]
        assert list(result["value"]) == [1, 2, 3]
        # Assert that the input DataFrame was not mutated
        assert list(df.columns) == ["value", "constant_num", "constant_text"]

    def test_drop_constant_columns_invalid_type_raises(self):
        with pytest.raises(
            TypeError, match="frame must be an ArFrame or a pandas DataFrame"
        ):
            ar.drop_constant_columns([1, 2, 3])

    def test_drop_constant_columns_zero_row_pandas_returns_new_object(self):
        df = pd.DataFrame({"a": pd.Series(dtype="int64")})

        result = ar.drop_constant_columns(df)

        assert result is not df
        assert result.shape == (0, 1)

    def test_drop_constant_columns_zero_row_arframe_returns_new_object(self):
        frame = ar.from_pandas(pd.DataFrame({"a": pd.Series(dtype="int64")}))

        result = ar.drop_constant_columns(frame)

        assert result is not frame
        assert result.shape == (0, 1)

    def test_drop_constant_columns_zero_row_attrs_not_shared(self):
        frame = ar.from_pandas(pd.DataFrame({"a": pd.Series(dtype="int64")}))

        frame._attrs = {"nested": {"x": 1}}

        result = ar.drop_constant_columns(frame)

        result._attrs["nested"]["x"] = 2

        assert frame._attrs["nested"]["x"] == 1


class TestDropEmptyColumns:
    def test_drop_empty_columns_removes_fully_empty_columns(self, tmp_path):
        csv_path = tmp_path / "drop_empty_columns.csv"
        csv_path.write_text(
            'all_null,all_blank,value\n,"",1\n,"   ",2\n,"",3\n',
            encoding="utf-8",
        )
        frame = ar.read_csv(csv_path)

        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)

        assert list(df.columns) == ["value"]
        assert list(df["value"]) == [1, 2, 3]

    def test_drop_empty_columns_keeps_partially_empty_columns(self, tmp_path):
        csv_path = tmp_path / "drop_empty_columns_partial.csv"
        csv_path.write_text(
            'maybe_empty,whitespace_then_value\n,"   "\n"",\nkept,x\n',
            encoding="utf-8",
        )
        frame = ar.read_csv(csv_path)

        result = ar.drop_empty_columns(frame)

        assert result.columns == frame.columns
        assert result.shape == frame.shape

    def test_drop_empty_columns_keeps_falsey_non_string_columns(self, tmp_path):
        csv_path = tmp_path / "drop_empty_columns_falsey.csv"
        csv_path.write_text(
            "zeros,string_zero\n0,0\n0,0\n0,0\n",
            encoding="utf-8",
        )
        frame = ar.read_csv(csv_path)

        result = ar.drop_empty_columns(frame)

        assert result.columns == ["zeros", "string_zero"]
        assert result.shape == frame.shape

    def test_drop_empty_columns_all_columns_dropped_preserves_row_count(self, tmp_path):
        csv_path = tmp_path / "drop_empty_columns_all.csv"
        csv_path.write_text('all_null,all_blank\n,""\n, \n', encoding="utf-8")
        frame = ar.read_csv(csv_path)

        result = ar.drop_empty_columns(frame)

        assert result.columns == []
        assert result.shape[1] == 0
        assert result.shape[0] in {0, 2}
        assert ar.to_pandas(result).shape[1] == 0

    def test_drop_empty_columns_preserves_schema_on_empty_frame(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        frame = ar.from_pandas(df)

        result = ar.drop_empty_columns(frame)

        assert result.columns == ["a", "b", "c"]
        assert result.shape[0] == 0
        assert result.shape[1] == 3

    def test_drop_empty_columns_zero_row_metadata_isolation(self):
        df = pd.DataFrame({"a": pd.Series(dtype="object")})
        frame = ar.from_pandas(df)
        frame._attrs = {"nested": {"x": 1}}

        result = ar.drop_empty_columns(frame)

        assert result is not frame
        assert result.columns == ["a"]
        assert result.shape == (0, 1)

        result._attrs["nested"]["x"] = 2
        assert frame._attrs["nested"]["x"] == 1


class TestClipNumeric:
    def test_clip_numeric_lower_only(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [-5, 0, 10]}))

        result = ar.clip_numeric(frame, lower=1)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [1, 1, 10]

    def test_clip_numeric_upper_only(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [-5, 0, 10]}))

        result = ar.clip_numeric(frame, upper=3)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [-5, 0, 3]

    def test_clip_numeric_both_bounds(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [-5, 2, 10]}))

        result = ar.clip_numeric(frame, lower=0, upper=5)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [0, 2, 5]

    def test_clip_numeric_all_numeric_subset_skips_non_numeric_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "value": [-5, 5, 20],
                    "label": ["low", "ok", "high"],
                }
            )
        )

        result = ar.clip_numeric(frame, lower=0, upper=10)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [0, 5, 10]
        assert list(df["label"]) == ["low", "ok", "high"]

    def test_clip_numeric_subset_only_requested_numeric_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "a": [-5, 0, 10],
                    "b": [-10, 5, 20],
                    "label": ["x", "y", "z"],
                }
            )
        )

        result = ar.clip_numeric(frame, lower=0, upper=8, subset=["b"])
        df = ar.to_pandas(result)

        assert list(df["a"]) == [-5, 0, 10]
        assert list(df["b"]) == [0, 5, 8]
        assert list(df["label"]) == ["x", "y", "z"]

    def test_clip_numeric_string_subset_rejected_before_native_execution(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "age": [1, 2]}))

        with pytest.raises(
            TypeError,
            match="subset must be a sequence of column names, not a string",
        ):
            ar.clip_numeric(frame, lower=0, subset="age")

    def test_clip_numeric_non_string_subset_item_rejected_before_native_execution(
        self,
    ):
        frame = ar.from_pandas(pd.DataFrame({"age": [1, 2, 3]}))

        with pytest.raises(
            TypeError,
            match="subset must contain only string column names",
        ):
            ar.clip_numeric(frame, lower=0, subset=[1])

    def test_clip_numeric_valid_tuple_subset_preserves_supported_behavior(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "a": [-5, 0, 10],
                    "b": [-10, 5, 20],
                    "label": ["x", "y", "z"],
                }
            )
        )

        result = ar.clip_numeric(frame, lower=0, upper=8, subset=("b",))
        df = ar.to_pandas(result)

        assert list(df["a"]) == [-5, 0, 10]
        assert list(df["b"]) == [0, 5, 8]
        assert list(df["label"]) == ["x", "y", "z"]

    def test_clip_numeric_keeps_missing_values(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [None, -5.0, 10.0]}))

        result = ar.clip_numeric(frame, lower=0, upper=5)
        df = ar.to_pandas(result)

        assert pd.isna(df["value"].iloc[0])
        assert list(df["value"].iloc[1:]) == [0.0, 5.0]

    def test_clip_numeric_unknown_subset_column_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.clip_numeric(frame, lower=0, subset=["missing"])

    def test_clip_numeric_non_numeric_subset_column_raises(self):
        frame = ar.from_pandas(
            pd.DataFrame({"value": [1, 2, 3], "label": ["x", "y", "z"]})
        )

        with pytest.raises(
            ValueError, match="clip_numeric only supports numeric columns"
        ):
            ar.clip_numeric(frame, lower=0, subset=["label"])

    def test_clip_numeric_no_bounds_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(
            ValueError, match="At least one of 'lower' or 'upper' must be provided"
        ):
            ar.clip_numeric(frame)

    def test_clip_numeric_inverted_bounds_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(ValueError, match="lower cannot be greater than upper"):
            ar.clip_numeric(frame, lower=5, upper=1)

    def test_clip_numeric_empty_subset_returns_frame_unchanged(self):
        # subset=[] must return the original frame without modification.
        # This was a previous review blocker; the guard lives in the Python wrapper
        # and must never reach the C++ layer.
        frame = ar.from_pandas(pd.DataFrame({"value": [-5, 0, 10]}))

        result = ar.clip_numeric(frame, lower=0, upper=5, subset=[])

        df_orig = ar.to_pandas(frame)
        df_result = ar.to_pandas(result)
        assert list(df_result["value"]) == list(df_orig["value"])

    def test_clip_numeric_non_integral_lower_on_int64_raises(self):
        # A float lower bound that is not integral (e.g. 1.5) must raise rather
        # than silently truncate to 1 via C++ static_cast<int64_t>.
        frame = ar.from_pandas(pd.DataFrame({"x": [0, 2, 5]}))

        with pytest.raises(ValueError, match="not an integer value"):
            ar.clip_numeric(frame, lower=1.5)

    def test_clip_numeric_non_integral_upper_on_int64_raises(self):
        # Same guard for the upper bound.
        frame = ar.from_pandas(pd.DataFrame({"x": [0, 2, 5]}))

        with pytest.raises(ValueError, match="not an integer value"):
            ar.clip_numeric(frame, upper=3.7)

    def test_clip_numeric_integral_float_bound_on_int64_accepted(self):
        # A float that is mathematically integral (e.g. 2.0) is fine.
        frame = ar.from_pandas(pd.DataFrame({"x": [-1, 2, 10]}))

        result = ar.clip_numeric(frame, lower=0.0, upper=5.0)
        df = ar.to_pandas(result)

        assert list(df["x"]) == [0, 2, 5]

    def test_clip_numeric_out_of_range_bound_on_int64_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"x": [-1, 2, 10]}))

        with pytest.raises(ValueError, match="within int64 range"):
            ar.clip_numeric(frame, upper=1e20)

        with pytest.raises(ValueError, match="within int64 range"):
            ar.clip_numeric(frame, lower=-1e20)

    def test_clip_numeric_non_integral_bound_on_float64_accepted(self):
        # Non-integral bounds are valid for float64 columns.
        frame = ar.from_pandas(pd.DataFrame({"v": [-1.0, 2.5, 9.9]}))

        result = ar.clip_numeric(frame, lower=1.5, upper=8.3)
        df = ar.to_pandas(result)

        assert list(df["v"]) == [1.5, 2.5, 8.3]

    def test_clip_numeric_rejects_bool_and_non_numeric_bounds(self):
        frame = ar.from_pandas(pd.DataFrame({"values": [1.0, 5.0, 10.0, 20.0]}))

        # Boolean bounds must be rejected (bool is subclass of int in Python,
        # so explicit rejection is required)
        with pytest.raises(TypeError, match="'lower' must be an int or float"):
            ar.clip_numeric(frame, lower=True)

        with pytest.raises(TypeError, match="'upper' must be an int or float"):
            ar.clip_numeric(frame, upper=False)

        with pytest.raises(TypeError, match="'lower' must be an int or float"):
            ar.clip_numeric(frame, lower="a")

        with pytest.raises(TypeError, match="'upper' must be an int or float"):
            ar.clip_numeric(frame, upper="10")

        # Valid int and float bounds must still work fine
        ar.clip_numeric(frame, lower=0, upper=15)
        ar.clip_numeric(frame, lower=0.5, upper=9.5)

    def test_clip_numeric_pipeline_rejects_invalid_bounds(self):
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))

        with pytest.raises(TypeError, match="'lower' must be an int or float"):
            ar.pipeline(
                frame,
                [("clip_numeric", {"lower": True})],
            )


class TestStandardizeMissingTokens:
    def test_normal_case(self):
        df = pd.DataFrame({"value": [1, 2, "N/A"]})
        result = ar.standardize_missing_tokens(df)
        assert isinstance(result, pd.DataFrame)
        assert pd.isna(result["value"].iloc[2])

    def test_normal_case_arframe(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, "N/A"]}))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        assert isinstance(result, ar.ArFrame)
        assert pd.isna(df["value"].iloc[2])

    def test_default_case(self):
        df = pd.DataFrame({"value": [1, 2, "-"]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["value"].iloc[2])

    def test_default_case_subset(self):
        df = pd.DataFrame(
            {
                "roll_no": ["001", "002", "003"],
                "name": ["Alice", "Bob", "Carter"],
                "marks": [100, 90, "-"],
            }
        )
        result = ar.standardize_missing_tokens(df, subset=["marks"])
        assert pd.isna(result["marks"].iloc[2])
        assert result["name"].iloc[2] == "Carter"

    def test_custom_case(self):
        df = pd.DataFrame({"value": [1, 2, "unknown"]})
        result = ar.standardize_missing_tokens(df, tokens=["unknown"])
        assert pd.isna(result["value"].iloc[2])

    def test_custom_case_subset(self):
        df = pd.DataFrame(
            {
                "roll_no": ["001", "002", "003"],
                "name": ["Alice", "Bob", "Carter"],
                "marks": [100, 90, "unknown"],
            }
        )
        result = ar.standardize_missing_tokens(df, tokens=["unknown"], subset=["marks"])
        assert pd.isna(result["marks"].iloc[2])
        assert result["name"].iloc[2] == "Carter"

    def test_non_string_columns(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = ar.standardize_missing_tokens(df)
        assert result["value"].iloc[0] == 1

    def test_unchanged_columns(self):
        df = pd.DataFrame({"value": [1, 2, "-"]})
        result = ar.standardize_missing_tokens(df, tokens=[])
        assert result["value"].iloc[2] == "-"

    def test_whitespace_only_values_remain_when_tokens_disabled(self):
        df = pd.DataFrame({"value": ["  ", "\t", "\n"]})

        result = ar.standardize_missing_tokens(df, tokens=[])

        assert result["value"].tolist() == ["  ", "\t", "\n"]

    def test_standardize_missing_tokens_unknown_subset_column_raises(self):
        frame = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.standardize_missing_tokens(frame, subset=["missing"])

    def test_standardize_missing_tokens_pandas_subset_returns_dataframe(self):
        df = pd.DataFrame({"name": ["N/A", "Alice"], "city": ["-", "Paris"]})

        result = ar.standardize_missing_tokens(df, subset=["name"])

        assert isinstance(result, pd.DataFrame)
        assert pd.isna(result.loc[0, "name"])
        assert result.loc[1, "name"] == "Alice"
        assert result["city"].tolist() == ["-", "Paris"]

    def test_standardize_missing_tokens_normalizes_whitespace_wrapped_defaults(self):
        df = pd.DataFrame({"value": ["NULL ", " NaN", "  ", "", "Alice "]})

        result = ar.standardize_missing_tokens(df)

        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert pd.isna(result["value"].iloc[2])
        assert pd.isna(result["value"].iloc[3])
        assert result["value"].iloc[4] == "Alice "

    def test_standardize_missing_tokens_normalizes_whitespace_wrapped_custom_tokens(
        self,
    ):
        df = pd.DataFrame(
            {
                "status": [" unknown ", "pending", " custom-null "],
                "note": [" untouched ", "unknown", "kept"],
            }
        )

        result = ar.standardize_missing_tokens(
            df, tokens=["unknown", "custom-null"], subset=["status"]
        )

        assert pd.isna(result["status"].iloc[0])
        assert result["status"].iloc[1] == "pending"
        assert pd.isna(result["status"].iloc[2])
        assert result["note"].tolist() == [" untouched ", "unknown", "kept"]

    def test_standardize_missing_tokens_normalizes_custom_token_list_entries(self):
        df = pd.DataFrame({"value": ["unknown", " Unknown ", "kept"]})

        result = ar.standardize_missing_tokens(df, tokens=["  UNKNOWN  "])

        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == "kept"

    def test_standardize_missing_tokens_normalizes_custom_token_list_entries_in_subset(
        self,
    ):
        df = pd.DataFrame(
            {
                "status": [" unknown ", "kept"],
                "note": ["UNKNOWN", "still here"],
            }
        )

        result = ar.standardize_missing_tokens(
            df, tokens=["  UNKNOWN  "], subset=["status"]
        )

        assert pd.isna(result["status"].iloc[0])
        assert result["status"].iloc[1] == "kept"
        assert result["note"].tolist() == ["UNKNOWN", "still here"]

    def test_standardize_missing_tokens_normalizes_tab_and_newline_wrapped_tokens(
        self,
    ):
        df = pd.DataFrame({"value": ["\tNULL\t", "\n NaN\n", "\t kept \n"]})

        result = ar.standardize_missing_tokens(df)

        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == "\t kept \n"

    def test_standardize_missing_tokens_subset_does_not_normalize_excluded_whitespace(
        self,
    ):
        df = pd.DataFrame(
            {
                "status": ["  ", "NULL "],
                "note": ["  ", "NULL "],
            }
        )

        result = ar.standardize_missing_tokens(df, subset=["status"])

        assert pd.isna(result["status"].iloc[0])
        assert pd.isna(result["status"].iloc[1])
        assert result["note"].tolist() == ["  ", "NULL "]

    def test_standardize_missing_tokens_normalizes_carriage_return_wrapped_tokens(
        self,
    ):
        df = pd.DataFrame({"value": ["\r\nNULL\r", "\r\n nAn \r\n", "\r kept \r"]})

        result = ar.standardize_missing_tokens(df)

        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == "\r kept \r"

    def test_standardize_missing_tokens_normalizes_nonbreaking_space_wrapped_tokens(
        self,
    ):
        nbsp = "\u00a0"
        df = pd.DataFrame({"value": [f"{nbsp}NULL{nbsp}", f"{nbsp} kept {nbsp}"]})

        result = ar.standardize_missing_tokens(df)

        assert pd.isna(result["value"].iloc[0])
        assert result["value"].iloc[1] == f"{nbsp} kept {nbsp}"

    def test_standardize_missing_tokens_custom_tokens_do_not_fall_back_to_defaults(
        self,
    ):
        df = pd.DataFrame({"value": [" NULL ", " custom-null ", "kept"]})

        result = ar.standardize_missing_tokens(df, tokens=["custom-null"])

        assert result["value"].iloc[0] == " NULL "
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == "kept"

    def test_standardize_missing_tokens_whitespace_only_custom_tokens_match_blank_values(
        self,
    ):
        df = pd.DataFrame({"value": ["  ", "\t", "\n", "kept"]})

        result = ar.standardize_missing_tokens(df, tokens=["   "])

        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert pd.isna(result["value"].iloc[2])
        assert result["value"].iloc[3] == "kept"

    def test_standardize_missing_tokens_preserves_existing_nulls_while_normalizing_wrapped_tokens(
        self,
    ):
        df = pd.DataFrame({"value": [None, pd.NA, " NULL ", "kept"]})

        result = ar.standardize_missing_tokens(df)

        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert pd.isna(result["value"].iloc[2])
        assert result["value"].iloc[3] == "kept"

    def test_standardize_missing_tokens_normalizes_wrapped_punctuation_defaults(self):
        df = pd.DataFrame({"value": [" ? ", "\t-\t", "--", "kept"]})

        result = ar.standardize_missing_tokens(df)

        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == "--"
        assert result["value"].iloc[3] == "kept"

    def test_standardize_missing_tokens_scalar_int_raises(self):
        frame = pd.DataFrame({"x": ["NA", "N", "ok"]})
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(frame, tokens=1)

    def test_standardize_missing_tokens_dict_raises(self):
        frame = pd.DataFrame({"x": ["NA", "N", "ok"]})
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(frame, tokens={"NA": "bad"})

    def test_standardize_missing_tokens_bare_string_raises(self):
        frame = pd.DataFrame({"x": ["NA", "N", "ok"]})
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(frame, tokens="NA")

    def test_standardize_missing_tokens_list_with_non_string_item_raises(self):
        frame = pd.DataFrame({"x": ["NA", "N", "ok"]})
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(frame, tokens=["NA", 1])


class TestDropConstantColumns:
    def test_drop_constant_columns_removes_constant_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "value": [1, 2, 3],
                    "constant_num": [7, 7, 7],
                    "constant_text": ["x", "x", "x"],
                }
            )
        )

        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)

        assert list(df.columns) == ["value"]
        assert list(df["value"]) == [1, 2, 3]

    def test_drop_constant_columns_keeps_non_constant_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "a": [1, 2, 1],
                    "b": ["x", "y", "x"],
                }
            )
        )

        result = ar.drop_constant_columns(frame)

        assert result.columns == frame.columns
        assert result.shape == frame.shape

    def test_drop_constant_columns_drops_all_null_column(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "all_null": [None, None],
                    "value": [1, 2],
                }
            )
        )

        result = ar.drop_constant_columns(frame)

        assert result.columns == ["value"]

    def test_drop_constant_columns_keeps_value_plus_null_column(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "maybe_constant": [1, 1, None],
                    "constant": [2, 2, 2],
                }
            )
        )

        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)

        assert list(df.columns) == ["maybe_constant"]
        assert df.shape == (3, 1)

    def test_drop_constant_columns_empty_frame_keeps_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "empty_num": pd.Series(dtype="float64"),
                    "empty_text": pd.Series(dtype="object"),
                }
            )
        )

        result = ar.drop_constant_columns(frame)

        assert result.columns == ["empty_num", "empty_text"]
        assert result.shape == frame.shape

    def test_drop_constant_columns_all_columns_dropped_reports_zero_rows(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": ["x"], "c": [None]}))

        result = ar.drop_constant_columns(frame)

        assert result.columns == []
        assert result.shape[0] == 1
        assert result.shape[1] == 0


class TestClipNumeric:
    def test_clip_numeric_lower_only(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [-5, 0, 10]}))

        result = ar.clip_numeric(frame, lower=1)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [1, 1, 10]

    def test_clip_numeric_upper_only(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [-5, 0, 10]}))

        result = ar.clip_numeric(frame, upper=3)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [-5, 0, 3]

    def test_clip_numeric_both_bounds(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [-5, 2, 10]}))

        result = ar.clip_numeric(frame, lower=0, upper=5)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [0, 2, 5]

    def test_clip_numeric_all_numeric_subset_skips_non_numeric_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "value": [-5, 5, 20],
                    "label": ["low", "ok", "high"],
                }
            )
        )

        result = ar.clip_numeric(frame, lower=0, upper=10)
        df = ar.to_pandas(result)

        assert list(df["value"]) == [0, 5, 10]
        assert list(df["label"]) == ["low", "ok", "high"]

    def test_clip_numeric_subset_only_requested_numeric_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "a": [-5, 0, 10],
                    "b": [-10, 5, 20],
                    "label": ["x", "y", "z"],
                }
            )
        )

        result = ar.clip_numeric(frame, lower=0, upper=8, subset=["b"])
        df = ar.to_pandas(result)

        assert list(df["a"]) == [-5, 0, 10]
        assert list(df["b"]) == [0, 5, 8]
        assert list(df["label"]) == ["x", "y", "z"]

    def test_clip_numeric_keeps_missing_values(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [None, -5.0, 10.0]}))

        result = ar.clip_numeric(frame, lower=0, upper=5)
        df = ar.to_pandas(result)

        assert pd.isna(df["value"].iloc[0])
        assert list(df["value"].iloc[1:]) == [0.0, 5.0]

    def test_clip_numeric_unknown_subset_column_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.clip_numeric(frame, lower=0, subset=["missing"])

    def test_clip_numeric_non_numeric_subset_column_raises(self):
        frame = ar.from_pandas(
            pd.DataFrame({"value": [1, 2, 3], "label": ["x", "y", "z"]})
        )

        with pytest.raises(
            ValueError, match="clip_numeric only supports numeric columns"
        ):
            ar.clip_numeric(frame, lower=0, subset=["label"])

    def test_clip_numeric_no_bounds_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(
            ValueError, match="At least one of 'lower' or 'upper' must be provided"
        ):
            ar.clip_numeric(frame)

    def test_clip_numeric_inverted_bounds_raises(self):
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))

        with pytest.raises(ValueError, match="lower cannot be greater than upper"):
            ar.clip_numeric(frame, lower=5, upper=1)


class TestStripWhitespace:
    def test_strip(self, csv_with_whitespace):
        frame = ar.read_csv(csv_with_whitespace)
        result = ar.strip_whitespace(frame)
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Alice"
        assert df["city"].iloc[1] == "London"

    def test_strip_subset(self, csv_with_whitespace):
        frame = ar.read_csv(csv_with_whitespace)
        result = ar.strip_whitespace(frame, subset=["name"])
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Alice"
        # city should still have whitespace

    def test_strip_tabs_and_newlines(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "name": ["\tAlice\n", "  Bob\t"],
                    "city": ["\nLondon ", "\tParis\t"],
                }
            )
        )

        result = ar.strip_whitespace(frame)

        df = ar.to_pandas(result)

        assert df["name"].tolist() == ["Alice", "Bob"]
        assert df["city"].tolist() == ["London", "Paris"]


class TestNormalizeCase:

    def test_lower(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.normalize_case(frame, subset=["name"], case_type="lower")

        df = ar.to_pandas(result)

        assert df["name"].iloc[0] == "alice"

    def test_upper(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.normalize_case(frame, subset=["name"], case_type="upper")

        df = ar.to_pandas(result)

        assert df["name"].iloc[0] == "ALICE"

    def test_title(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.normalize_case(frame, subset=["name"], case_type="title")

        df = ar.to_pandas(result)

        assert df["name"].iloc[0] == "Alice"

    def test_title_hyphen(self):
        import pandas as pd

        frame = ar.from_pandas(
            pd.DataFrame({"name": ["hello-world", "jean-luc picard"]})
        )
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Hello-World"
        assert df["name"].iloc[1] == "Jean-Luc Picard"

    def test_title_underscore(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"name": ["hello_world", "foo_bar_baz"]}))
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Hello_World"
        assert df["name"].iloc[1] == "Foo_Bar_Baz"

    def test_title_period(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"name": ["dr.strange", "mr.smith"]}))
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Dr.Strange"
        assert df["name"].iloc[1] == "Mr.Smith"

    def test_title_slash(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"name": ["hello/world", "foo/bar"]}))
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Hello/World"
        assert df["name"].iloc[1] == "Foo/Bar"

    def test_unicode_bytes_are_preserved_for_lower_and_upper(self):
        import pandas as pd

        frame = ar.from_pandas(
            pd.DataFrame({"city": ["São Paulo", "München", "東京", "Dev 🚀"]})
        )

        lower = ar.to_pandas(
            ar.normalize_case(frame, subset=["city"], case_type="lower")
        )
        upper = ar.to_pandas(
            ar.normalize_case(frame, subset=["city"], case_type="upper")
        )

        assert lower["city"].tolist() == ["são paulo", "münchen", "東京", "dev 🚀"]
        assert upper["city"].tolist() == ["SãO PAULO", "MüNCHEN", "東京", "DEV 🚀"]

    def test_unicode_bytes_are_preserved_for_title(self):
        import pandas as pd

        frame = ar.from_pandas(
            pd.DataFrame({"city": ["são-paulo", "münchen central", "東京 station"]})
        )

        result = ar.normalize_case(frame, subset=["city"], case_type="title")
        df = ar.to_pandas(result)

        assert df["city"].tolist() == ["São-Paulo", "München Central", "東京 Station"]

    def test_title_preserves_non_ascii_word_prefixes(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"word": ["éclair", "ñandú", "über-cool"]}))

        result = ar.normalize_case(frame, subset=["word"], case_type="title")
        df = ar.to_pandas(result)

        assert df["word"].tolist() == ["éclair", "ñandú", "über-Cool"]

    def test_invalid_case_type_int(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"x": ["A"]}))
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(frame, case_type=123)

    def test_invalid_case_type_none(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"x": ["A"]}))
        with pytest.raises(TypeError, match="case_type must be a string"):
            ar.normalize_case(frame, case_type=None)

    def test_invalid_case_type_string(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"x": ["A"]}))
        with pytest.raises(ValueError, match="case_type must be one of"):
            ar.normalize_case(frame, case_type="invalid")


class TestNormalizeUnicode:
    def test_normalize_unicode(self):
        import unicodedata

        import pandas as pd

        import arnio as ar

        df = pd.DataFrame({"text": ["cafe\u0301"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFC")
        result_df = ar.to_pandas(result)
        assert result_df["text"].iloc[0] == unicodedata.normalize("NFC", "cafe\u0301")

    def test_normalize_unicode_non_string_form_raises(self):
        import pandas as pd
        import pytest

        import arnio as ar

        df = pd.DataFrame({"text": ["hello"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="form must be a string"):
            ar.normalize_unicode(frame, form=["NFC"])

    def test_normalize_unicode_no_pandas_roundtrip(self):
        import pandas as pd

        import arnio as ar
        import arnio.convert as convert_mod

        df = pd.DataFrame({"text": ["café", "naïve"]})
        frame = ar.from_pandas(df)
        original = convert_mod.to_pandas

        def _should_not_be_called(*a, **kw):
            raise AssertionError("normalize_unicode called to_pandas!")

        convert_mod.to_pandas = _should_not_be_called
        try:
            result = ar.normalize_unicode(frame)
        finally:
            convert_mod.to_pandas = original

        result_df = original(result)
        assert result_df["text"].tolist() == ["café", "naïve"]

    def test_normalize_unicode_nfd_form(self):
        import unicodedata

        import pandas as pd

        import arnio as ar

        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFD")
        result_df = ar.to_pandas(result)
        assert (
            unicodedata.normalize("NFD", result_df["text"].iloc[0])
            == result_df["text"].iloc[0]
        )

    def test_normalize_unicode_nfkc_form(self):
        import pandas as pd

        import arnio as ar

        df = pd.DataFrame({"text": ["ﬁle"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, form="NFKC")
        result_df = ar.to_pandas(result)
        assert result_df["text"].iloc[0] == "file"

    def test_normalize_unicode_preserves_nulls(self):
        import pandas as pd

        import arnio as ar

        df = pd.DataFrame({"text": ["café", None, "naïve"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"].iloc[0] == "café"
        assert pd.isna(result_df["text"].iloc[1])
        assert result_df["text"].iloc[2] == "naïve"

    def test_normalize_unicode_non_string_columns_unchanged(self):
        import pandas as pd

        import arnio as ar

        df = pd.DataFrame({"text": ["café"], "score": [42], "flag": [True]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame)
        result_df = ar.to_pandas(result)
        assert result_df["score"].iloc[0] == 42
        assert (
            result_df["flag"].iloc[0] is True
            or result_df["flag"].iloc[0] == True  # noqa: E712
        )

    def test_normalize_unicode_subset_only_targets_specified_columns(self):
        import pandas as pd

        import arnio as ar

        raw_a = "café"
        raw_b = "résumé"
        df = pd.DataFrame({"a": [raw_a], "b": [raw_b]})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame, subset=["a"])
        result_df = ar.to_pandas(result)
        assert result_df["a"].iloc[0] == "café"
        assert result_df["b"].iloc[0] == raw_b

    def test_normalize_unicode_invalid_form_raises(self):
        import pandas as pd
        import pytest

        import arnio as ar

        df = pd.DataFrame({"text": ["hello"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="Unsupported Unicode normalization form"):
            ar.normalize_unicode(frame, form="XYZ")

    def test_normalize_unicode_large_frame_no_pandas(self):
        import pandas as pd

        import arnio as ar

        n = 10_000
        df = pd.DataFrame({"text": ["café"] * n, "other": list(range(n))})
        frame = ar.from_pandas(df)
        result = ar.normalize_unicode(frame)
        result_df = ar.to_pandas(result)
        assert all(v == "café" for v in result_df["text"])

    def test_normalize_unicode_attrs_deepcopy(self):
        import pandas as pd

        import arnio as ar

        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        frame._attrs = {"meta": {"key": "value"}}
        result = ar.normalize_unicode(frame)
        result._attrs["meta"]["key"] = "mutated"
        assert frame._attrs["meta"]["key"] == "value"

    def test_normalize_unicode_zero_columns(self):
        import pandas as pd

        import arnio as ar

        # Non-empty zero-column frame
        frame_3_0 = ar.from_pandas(pd.DataFrame(index=range(3)))
        assert frame_3_0.shape == (3, 0)
        result_3_0 = ar.normalize_unicode(frame_3_0)
        assert result_3_0.shape == (3, 0)

        # Empty zero-column frame
        frame_0_0 = ar.from_pandas(pd.DataFrame())
        assert frame_0_0.shape == (0, 0)
        result_0_0 = ar.normalize_unicode(frame_0_0)
        assert result_0_0.shape == (0, 0)

        # Normal string-column behavior
        df_normal = pd.DataFrame({"text": ["cafe\u0301"], "other": [1]})
        frame_normal = ar.from_pandas(df_normal)
        result_normal = ar.normalize_unicode(frame_normal)
        assert result_normal.shape == (1, 2)
        assert ar.to_pandas(result_normal)["text"].iloc[0] == "café"

        # attrs preservation on the zero-column path
        frame_3_0_attrs = ar.from_pandas(pd.DataFrame(index=range(3)))
        frame_3_0_attrs._attrs = {"key": "value"}
        result_3_0_attrs = ar.normalize_unicode(frame_3_0_attrs)
        assert result_3_0_attrs.shape == (3, 0)
        assert result_3_0_attrs._attrs == {"key": "value"}

        # attrs deepcopy check on zero-column path
        result_3_0_attrs._attrs["key"] = "mutated"
        assert frame_3_0_attrs._attrs["key"] == "value"


class TestAttrsPreservation:
    """Native cleaning wrappers must carry over ArFrame._attrs via deep copy."""

    def _base_frame(self):
        import pandas as pd

        df = pd.DataFrame(
            {"name": [" Alice ", " Bob "], "age": [20, 30], "score": [1.5, 2.5]}
        )
        frame = ar.from_pandas(df)
        frame._attrs = {"source": "crm", "meta": {"version": 1}}
        return frame

    @pytest.mark.parametrize(
        "op_name, fn",
        [
            ("drop_nulls", lambda f: ar.drop_nulls(f)),
            ("fill_nulls", lambda f: ar.fill_nulls(f, 0)),
            ("drop_duplicates", lambda f: ar.drop_duplicates(f)),
            ("strip_whitespace", lambda f: ar.strip_whitespace(f)),
            (
                "normalize_case",
                lambda f: ar.normalize_case(f, subset=["name"], case_type="lower"),
            ),
            (
                "clip_numeric",
                lambda f: ar.clip_numeric(f, subset=["age"], lower=0, upper=99),
            ),
            ("rename_columns", lambda f: ar.rename_columns(f, {"score": "score2"})),
            ("trim_column_names", lambda f: ar.trim_column_names(f)),
            ("cast_types", lambda f: ar.cast_types(f, {"age": "float64"})),
            ("normalize_unicode", lambda f: ar.normalize_unicode(f, subset=["name"])),
        ],
    )
    def test_attrs_propagated(self, op_name, fn):
        frame = self._base_frame()
        result = fn(frame)
        assert result._attrs == {
            "source": "crm",
            "meta": {"version": 1},
        }, f"{op_name} dropped _attrs"

    @pytest.mark.parametrize(
        "op_name, fn",
        [
            ("drop_nulls", lambda f: ar.drop_nulls(f)),
            ("fill_nulls", lambda f: ar.fill_nulls(f, 0)),
            ("drop_duplicates", lambda f: ar.drop_duplicates(f)),
            ("strip_whitespace", lambda f: ar.strip_whitespace(f)),
            (
                "normalize_case",
                lambda f: ar.normalize_case(f, subset=["name"], case_type="lower"),
            ),
            (
                "clip_numeric",
                lambda f: ar.clip_numeric(f, subset=["age"], lower=0, upper=99),
            ),
            ("rename_columns", lambda f: ar.rename_columns(f, {"score": "score2"})),
            ("trim_column_names", lambda f: ar.trim_column_names(f)),
            ("cast_types", lambda f: ar.cast_types(f, {"age": "float64"})),
            ("normalize_unicode", lambda f: ar.normalize_unicode(f, subset=["name"])),
        ],
    )
    def test_attrs_deep_copy_isolated(self, op_name, fn):
        frame = self._base_frame()
        result = fn(frame)
        result._attrs["meta"]["version"] = 999
        assert (
            frame._attrs["meta"]["version"] == 1
        ), f"{op_name} shared _attrs by reference instead of deep copying"

    def test_drop_duplicates_zero_columns_preserves_attrs(self):
        """drop_duplicates zero-column early return must propagate attrs."""
        import pandas as pd

        from arnio._core import _Frame

        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        # Build a genuine zero-column frame with rows intact
        frame._frame = _Frame.from_dict({}, {}, 3)
        frame._attrs = {"source": "crm", "meta": {"version": 1}}
        assert frame.shape == (3, 0)

        result = ar.drop_duplicates(frame)

        assert result._attrs == {
            "source": "crm",
            "meta": {"version": 1},
        }, "drop_duplicates zero-column path dropped _attrs"

    def test_drop_duplicates_zero_columns_attrs_deep_copy_isolated(self):
        """drop_duplicates zero-column result attrs must be a deep copy."""
        import pandas as pd

        from arnio._core import _Frame

        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        frame._frame = _Frame.from_dict({}, {}, 3)
        frame._attrs = {"source": "crm", "meta": {"version": 1}}

        result = ar.drop_duplicates(frame)
        result._attrs["meta"]["version"] = 999

        assert (
            frame._attrs["meta"]["version"] == 1
        ), "drop_duplicates zero-column path shared _attrs by reference instead of deep copying"

    def test_empty_attrs_not_propagated(self):
        """When source frame has no attrs, result attrs should also be empty."""
        frame = ar.from_pandas(
            __import__("pandas").DataFrame({"name": ["Alice"], "age": [20]})
        )
        assert frame._attrs == {}
        result = ar.strip_whitespace(frame)
        assert result._attrs == {}


class TestParseBoolStrings:
    def test_parse_basic_bool_strings(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", "no", "True", "0"],
            }
        )

        frame = ar.from_pandas(df)

        result = ar.pipeline(
            frame,
            [
                ("parse_bool_strings",),
            ],
        )

        cleaned = ar.to_pandas(result)

        assert cleaned["active"].tolist() == [True, False, True, False]

    def test_parse_bool_strings_preserves_unknown_values(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", "maybe", "0"],
            }
        )

        frame = ar.from_pandas(df)

        result = ar.pipeline(
            frame,
            [
                ("parse_bool_strings",),
            ],
        )

        cleaned = ar.to_pandas(result)

        assert cleaned["active"].tolist() == [
            "True",
            "maybe",
            "False",
        ]

    def test_parse_bool_strings_mixed_object_column(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", 123, "0"],
            },
            dtype=object,
        )

        frame = ar.from_pandas(df)

        result = ar.pipeline(
            frame,
            [
                ("parse_bool_strings",),
            ],
        )

        cleaned = ar.to_pandas(result)

        assert cleaned["active"].tolist() == [
            "True",
            "123",
            "False",
        ]

    def test_parse_bool_strings_direct_usage(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": [" YES ", "no", "maybe", None],
            }
        )

        frame = ar.from_pandas(df)

        result = ar.parse_bool_strings(frame)

        cleaned = ar.to_pandas(result)

        assert cleaned["active"].tolist()[:3] == [
            "True",
            "False",
            "maybe",
        ]

        assert pd.isna(cleaned["active"].iloc[3])

    def test_parse_bool_strings_subset(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", "no"],
                "other": ["YES", "no"],
            },
            dtype=object,
        )

        frame = ar.from_pandas(df)

        result = ar.parse_bool_strings(
            frame,
            subset=["active"],
        )

        cleaned = ar.to_pandas(result)

        assert cleaned["active"].tolist() == [True, False]
        assert cleaned["other"].tolist() == ["YES", "no"]

    def test_parse_bool_strings_subset_skips_existing_bool_columns(self):
        import pandas as pd

        import arnio as ar

        df = pd.DataFrame(
            {
                "flag": [True, False, True],
            }
        )

        frame = ar.from_pandas(df)

        result = ar.parse_bool_strings(
            frame,
            subset=["flag"],
        )

        result_df = ar.to_pandas(result)

        assert result_df["flag"].tolist() == [True, False, True]
        assert str(result_df["flag"].dtype) == "boolean"

    def test_parse_bool_strings_custom_values(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "status": [
                    "enabled",
                    "disabled",
                    " ENABLED ",
                    " DISABLED ",
                    "maybe",
                ],
            },
            dtype=object,
        )

        frame = ar.from_pandas(df)

        result = ar.parse_bool_strings(
            frame,
            true_values={"enabled"},
            false_values={"disabled"},
        )

        cleaned = ar.to_pandas(result)

        assert cleaned["status"].tolist() == [
            "True",
            "False",
            "True",
            "False",
            "maybe",
        ]

    def test_parse_bool_strings_overlap_rejected(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["yes", "no"],
            },
            dtype=object,
        )

        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            ar.parse_bool_strings(
                frame,
                true_values={"yes"},
                false_values={" YES "},
            )

    def test_parse_bool_strings_invalid_subset_type(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", "no"],
            }
        )

        frame = ar.from_pandas(df)

        with pytest.raises(TypeError):
            ar.parse_bool_strings(frame, subset="active")

    def test_parse_bool_strings_empty_subset(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", "no"],
            }
        )

        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            ar.parse_bool_strings(frame, subset=[])

    def test_parse_bool_strings_accepts_tuple_subset(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", "no"],
                "name": ["Alice", "Bob"],
            }
        )

        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=("active",))
        out = ar.to_pandas(result)

        assert out["active"].tolist() == [True, False]
        assert out["name"].tolist() == ["Alice", "Bob"]

    def test_parse_bool_strings_missing_subset_column(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "active": ["YES", "no"],
            }
        )

        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            ar.parse_bool_strings(frame, subset=["missing"])

    def test_parse_bool_strings_non_string_true_values_raises(self):
        """Regression: non-string items in true_values must raise TypeError,
        not crash with AttributeError on .strip().lower()."""
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="true_values must contain only strings"):
            ar.parse_bool_strings(frame, true_values={1, "yes"})

    def test_parse_bool_strings_non_string_false_values_raises(self):
        """Regression: non-string items in false_values must raise TypeError,
        not crash with AttributeError on .strip().lower()."""
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="false_values must contain only strings"):
            ar.parse_bool_strings(frame, false_values={0, "no"})

    def test_parse_bool_strings_none_in_custom_values_raises(self):
        """Regression: None in true_values/false_values must raise TypeError."""
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="true_values must contain only strings"):
            ar.parse_bool_strings(frame, true_values={"yes", None})

    def test_parse_bool_strings_bool_in_custom_values_raises(self):
        """Regression: bool items in true_values must raise TypeError."""
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="true_values must contain only strings"):
            ar.parse_bool_strings(frame, true_values={True, "yes"})

    def test_parse_bool_strings_other_non_string_types_in_custom_values_raises(self):
        """Test that custom sets containing floats, ints, or None raise TypeError."""
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        # Float
        with pytest.raises(
            TypeError, match="true_values must contain only strings, got float"
        ):
            ar.parse_bool_strings(frame, true_values={3.14, "yes"})

        with pytest.raises(
            TypeError, match="false_values must contain only strings, got float"
        ):
            ar.parse_bool_strings(frame, false_values={1.5, "no"})

        # Int
        with pytest.raises(
            TypeError, match="true_values must contain only strings, got int"
        ):
            ar.parse_bool_strings(frame, true_values={42, "yes"})

        # NoneType
        with pytest.raises(
            TypeError, match="true_values must contain only strings, got NoneType"
        ):
            ar.parse_bool_strings(frame, true_values={None, "yes"})

    def test_parse_bool_strings_non_iterable_custom_values_raises(self):
        """Test that passing a completely non-iterable type (like int, float, bool) to true_values/false_values raises TypeError."""
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        with pytest.raises(
            TypeError, match="true_values must be a set, list, or tuple of strings"
        ):
            ar.parse_bool_strings(frame, true_values=123)

        with pytest.raises(
            TypeError,
            match="false_values must be a set, list, or tuple of strings",
        ):
            ar.parse_bool_strings(frame, false_values=45.6)

    def test_parse_bool_strings_rejects_mapping_containers(self):
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        with pytest.raises(
            TypeError,
            match="true_values must be a set, list, or tuple of strings",
        ):
            ar.parse_bool_strings(frame, true_values={"yes": 1})

        with pytest.raises(
            TypeError,
            match="false_values must be a set, list, or tuple of strings",
        ):
            ar.parse_bool_strings(frame, false_values={"no": 1})

    def test_parse_bool_strings_overlap_whitespace_and_case_normalization(self):
        """Test that tokens that overlap after case folding and whitespace stripping are correctly rejected."""
        import pandas as pd

        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        # Exact overlap
        with pytest.raises(ValueError, match="overlap after normalization: {'yes'}"):
            ar.parse_bool_strings(frame, true_values={"yes"}, false_values={"yes"})

        # Overlap after whitespace stripping and case folding
        with pytest.raises(ValueError, match="overlap after normalization: {'yes'}"):
            ar.parse_bool_strings(frame, true_values={" YES "}, false_values={"yes"})

        with pytest.raises(ValueError, match="overlap after normalization: {'yes'}"):
            ar.parse_bool_strings(frame, true_values={"yes"}, false_values={"Yes"})

    def test_parse_bool_strings_empty_custom_values_sets(self):
        """Test that empty custom true_values and false_values sets are accepted and behave as no-ops for matching."""
        import pandas as pd

        df = pd.DataFrame({"active": ["true", "false", "yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        # Empty true_values means no values are converted to True
        result1 = ar.parse_bool_strings(frame, true_values=set())
        cleaned1 = ar.to_pandas(result1)
        assert cleaned1["active"].tolist() == ["true", "False", "yes", "False"]

        # Empty false_values means no values are converted to False
        result2 = ar.parse_bool_strings(frame, false_values=set())
        cleaned2 = ar.to_pandas(result2)
        assert cleaned2["active"].tolist() == ["True", "false", "True", "no"]

    def test_parse_bool_strings_rejects_bare_strings(self):
        df = pd.DataFrame({"active": ["yes", "no"]}, dtype=object)
        frame = ar.from_pandas(df)

        with pytest.raises(
            TypeError,
            match="true_values must be a set/list/tuple of strings, not a bare string",
        ):
            ar.parse_bool_strings(frame, true_values="yes")

        with pytest.raises(
            TypeError,
            match="false_values must be a set/list/tuple of strings, not a bare string",
        ):
            ar.parse_bool_strings(frame, false_values="no")

        with pytest.raises(
            TypeError,
            match="true_values must be a set/list/tuple of strings, not a bare string",
        ):
            ar.parse_bool_strings(frame, true_values=b"yes")

        with pytest.raises(
            TypeError,
            match="false_values must be a set/list/tuple of strings, not a bare string",
        ):
            ar.parse_bool_strings(frame, false_values=b"no")

    def test_parse_bool_strings_implicit_empty_and_whitespace(self):
        """
        Test that implicit empty strings, whitespace-only strings, and unsupported
        tokens are preserved completely unchanged, as per current design contracts.
        """
        import pandas as pd

        import arnio as ar

        # Scenario 1: Testing Default Tokens (Standard behavior)
        raw_data_default = {
            "bool_col": ["True", "False", "", "   ", "unsupported_token"]
        }
        frame_default = ar.from_pandas(pd.DataFrame(raw_data_default))

        result_default = ar.parse_bool_strings(frame_default)
        df_default = ar.to_pandas(result_default)

        # Checking that empty/whitespace strings are strictly preserved unchanged
        assert df_default["bool_col"].iloc[2] == ""
        assert df_default["bool_col"].iloc[3] == "   "
        assert df_default["bool_col"].iloc[4] == "unsupported_token"

        # Scenario 2: Testing Custom Tokens (As requested by the maintainer)
        raw_data_custom = {"custom_col": ["yea", "nay", "", "   "]}
        frame_custom = ar.from_pandas(pd.DataFrame(raw_data_custom))

        result_custom = ar.parse_bool_strings(
            frame_custom, true_values=["yea"], false_values=["nay"]
        )
        df_custom = ar.to_pandas(result_custom)

        # Checking that for custom tokens, empty/whitespace strings are still completely untouched
        assert df_custom["custom_col"].iloc[2] == ""
        assert df_custom["custom_col"].iloc[3] == "   "

        # Verification that parsing action occurred perfectly for all values
        assert df_custom["custom_col"].to_list() == ["True", "False", "", "   "]


class TestRemoveControlCharacters:

    def test_remove_control_characters(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        result = ar.remove_control_characters(frame, subset=["name"])

        df = ar.to_pandas(result)

        assert isinstance(df["name"].iloc[0], str)
        assert "\n" not in df["name"].iloc[0]
        assert "\t" not in df["name"].iloc[0]

    def test_title_hyphen(self):
        import pandas as pd

        frame = ar.from_pandas(
            pd.DataFrame({"name": ["hello-world", "jean-luc picard"]})
        )
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Hello-World"
        assert df["name"].iloc[1] == "Jean-Luc Picard"

    def test_title_underscore(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"name": ["hello_world", "foo_bar_baz"]}))
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Hello_World"
        assert df["name"].iloc[1] == "Foo_Bar_Baz"

    def test_title_period(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"name": ["dr.strange", "mr.smith"]}))
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Dr.Strange"
        assert df["name"].iloc[1] == "Mr.Smith"

    def test_title_slash(self):
        import pandas as pd

        frame = ar.from_pandas(pd.DataFrame({"name": ["hello/world", "foo/bar"]}))
        result = ar.normalize_case(frame, subset=["name"], case_type="title")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Hello/World"
        assert df["name"].iloc[1] == "Foo/Bar"


class TestRenameColumns:
    def test_rename(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        result = ar.rename_columns(frame, {"name": "full_name", "age": "years"})
        assert "full_name" in result.columns
        assert "years" in result.columns
        assert "name" not in result.columns

    def test_rename_rejects_non_mapping(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(
            TypeError, match="mapping must be a mapping of string keys to strings"
        ):
            ar.rename_columns(frame, [("name", "full_name")])

    def test_rename_rejects_non_string_target(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="values must be non-empty strings"):
            ar.rename_columns(frame, {"name": 123})

    def test_rename_rejects_duplicate_targets(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(ValueError, match="target names would create duplicates"):
            ar.rename_columns(frame, {"name": "person", "age": "person"})

    def test_rename_rejects_collision_with_unmapped_column(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(ValueError, match="collide with existing columns"):
            ar.rename_columns(frame, {"name": "age"})

    def test_rename_columns_rejects_empty_target(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="non-empty strings"):
            ar.rename_columns(frame, {"name": ""})

    def test_rename_columns_rejects_whitespace_target(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(
            TypeError,
            match="values must be non-empty strings",
        ):
            ar.rename_columns(frame, {"name": "   "})

    # --- Regression tests for non-dict mapping validation (bug fix) ---

    def test_rename_rejects_none_with_clear_type_error(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(TypeError, match="must be a mapping.*'NoneType'"):
            ar.rename_columns(frame, None)

    def test_rename_rejects_list_of_tuples_with_clear_type_error(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(TypeError, match="must be a mapping.*'list'"):
            ar.rename_columns(frame, [("name", "full_name")])

    def test_rename_rejects_integer_with_clear_type_error(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(TypeError, match="must be a mapping.*'int'"):
            ar.rename_columns(frame, 42)

    def test_rename_rejects_string_with_clear_type_error(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(TypeError, match="must be a mapping.*'str'"):
            ar.rename_columns(frame, "name:full_name")

    def test_rename_valid_dict_still_works(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        result = ar.rename_columns(frame, {"name": "full_name"})
        assert "full_name" in result.columns
        assert "name" not in result.columns

    def test_rename_rejects_non_string_key(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(
            TypeError, match="keys must contain only string column names"
        ):
            ar.rename_columns(frame, {123: "new_name"})


class TestTrimColumnNames:
    def test_trim_column_names_basic(self):
        df = pd.DataFrame({" name ": [1], " age ": [2]})
        frame = from_pandas(df)
        result = ar.trim_column_names(frame)
        assert to_pandas(result).columns.tolist() == ["name", "age"]

    def test_trim_column_names_rejects_non_frame_input(self):
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.trim_column_names([])

    def test_trim_column_names_already_clean(self):
        df = pd.DataFrame({"name": [1], "age": [2]})
        frame = from_pandas(df)
        result = ar.trim_column_names(frame)
        assert to_pandas(result).columns.tolist() == ["name", "age"]

    def test_trim_column_names_mixed(self):
        df = pd.DataFrame({" name": [1], "age ": [2], "score": [3]})
        frame = from_pandas(df)
        result = ar.trim_column_names(frame)
        assert to_pandas(result).columns.tolist() == ["name", "age", "score"]

    def test_trim_column_names_preserves_order(self):
        df = pd.DataFrame({" c ": [1], " b ": [2], " a ": [3]})
        frame = from_pandas(df)
        result = ar.trim_column_names(frame)
        assert to_pandas(result).columns.tolist() == ["c", "b", "a"]

    def test_trim_column_names_duplicate_raises(self):
        df = pd.DataFrame({" name": [1], "name ": [2]})
        frame = from_pandas(df)
        with pytest.raises(ValueError, match="duplicates"):
            ar.trim_column_names(frame)

    def test_trim_column_names_whitespace_only(self):
        df = pd.DataFrame({"   ": [1], " b ": [2]})
        frame = from_pandas(df)
        result = ar.trim_column_names(frame)
        assert to_pandas(result).columns.tolist() == ["", "b"]

    def test_trim_column_names_skips_pandas_round_trip(self, monkeypatch):
        import arnio.convert as convert

        def _boom(*_args, **_kwargs):
            raise AssertionError("trim_column_names should not call to_pandas")

        monkeypatch.setattr(convert, "to_pandas", _boom)
        frame = from_pandas(pd.DataFrame({" name ": [1]}))
        result = ar.trim_column_names(frame)
        assert result.columns == ["name"]


class TestMixedFrameValidation:
    @pytest.mark.parametrize(
        ("func", "kwargs"),
        [
            (
                "combine_columns",
                {"subset": ["a"], "separator": "-", "output_column": "combined"},
            ),
            ("drop_constant_columns", {}),
        ],
    )
    def test_mixed_helpers_reject_non_frame_input(self, func, kwargs):
        with pytest.raises(
            TypeError, match="frame must be an ArFrame or a pandas DataFrame"
        ):
            getattr(ar, func)([], **kwargs)


def test_from_pandas_multiindex_columns_are_stringified():
    df = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("a", "x"),
                ("b", "y"),
            ]
        ),
    )

    frame = ar.from_pandas(df)

    result = ar.to_pandas(frame)

    assert list(result.columns) == ["('a', 'x')", "('b', 'y')"]
    assert not isinstance(result.columns, pd.MultiIndex)

    assert result.iloc[0].tolist() == [1, 2]


class TestCastTypes:
    def test_cast_int_to_string(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        result = ar.cast_types(frame, {"age": "string"})
        assert result.dtypes["age"] == "string"

    def test_cast_int_to_float(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        result = ar.cast_types(frame, {"age": "float64"})
        assert result.dtypes["age"] == "float64"

    def test_cast_unknown_type_rejected(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(ar.TypeCastError, match="Unknown target dtype"):
            ar.cast_types(frame, {"age": "decimal"})

    def test_cast_invalid_value_raises_by_default(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["1", "bad"]}))

        with pytest.raises(ar.TypeCastError, match="Cannot cast column 'age'"):
            ar.cast_types(frame, {"age": "int64"})

    def test_cast_invalid_value_can_be_coerced(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["1", "bad"]}))

        result = ar.cast_types(frame, {"age": "int64"}, errors="coerce")
        df = ar.to_pandas(result)

        assert result.dtypes["age"] == "int64"
        assert df["age"].iloc[0] == 1
        assert pd.isna(df["age"].iloc[1])

    def test_cast_rejects_invalid_errors_policy(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(ValueError, match="errors must be one of"):
            ar.cast_types(frame, {"age": "int64"}, errors="warn")

    @pytest.mark.parametrize(
        "mapping",
        [
            None,
            [("age", "int64")],
            (("age", "int64"),),
            "age=int64",
        ],
    )
    def test_cast_rejects_non_mapping_with_clear_error(self, sample_csv, mapping):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(
            TypeError, match="mapping must be a mapping of string keys to strings"
        ):
            ar.cast_types(frame, mapping)

    def test_cast_bool_rejects_unknown_strings(self):
        frame = ar.from_pandas(pd.DataFrame({"active": ["true", "maybe"]}))

        with pytest.raises(ar.TypeCastError, match="Cannot cast column 'active'"):
            ar.cast_types(frame, {"active": "bool"})

    def test_cast_int_to_string_value_correctness(self, sample_csv):
        # Checks actual values, not just dtype
        frame = ar.read_csv(sample_csv)
        result = ar.cast_types(frame, {"age": "string"})
        df = ar.to_pandas(result)
        assert list(df["age"]) == ["30", "25", "35"]

    def test_cast_int_to_float_value_correctness(self, sample_csv):
        # Checks actual values, not just dtype
        frame = ar.read_csv(sample_csv)
        result = ar.cast_types(frame, {"age": "float64"})
        df = ar.to_pandas(result)
        assert list(df["age"]) == [30.0, 25.0, 35.0]

    def test_cast_float_to_int_raises_by_default(self):
        # float→int is lossy, so it raises TypeCastError by default
        frame = ar.from_pandas(pd.DataFrame({"score": [3.7, 2.1, 1.9]}))
        with pytest.raises(ar.TypeCastError, match="Cannot cast column 'score'"):
            ar.cast_types(frame, {"score": "int64"})

    def test_cast_float_to_int_coerces_to_null(self):
        # with errors="coerce", unparseable floats become null
        frame = ar.from_pandas(pd.DataFrame({"score": [3.7, 2.1]}))
        result = ar.cast_types(frame, {"score": "int64"}, errors="coerce")
        df = ar.to_pandas(result)
        assert result.dtypes["score"] == "int64"
        assert pd.isna(df["score"].iloc[0])

    def test_cast_null_preserved_through_int_to_float(self):
        # Nulls must survive type conversion
        frame = ar.from_pandas(
            pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64")})
        )
        result = ar.cast_types(frame, {"x": "float64"})
        df = ar.to_pandas(result)
        assert result.dtypes["x"] == "float64"
        assert df["x"].iloc[0] == 1.0
        assert pd.isna(df["x"].iloc[1])
        assert df["x"].iloc[2] == 3.0

    def test_cast_null_preserved_through_string_to_int_coerce(self):
        # Nulls in string column stay null after coerce cast
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", None, "30"]}))
        result = ar.cast_types(frame, {"age": "int64"}, errors="coerce")
        df = ar.to_pandas(result)
        assert pd.isna(df["age"].iloc[1])
        assert df["age"].iloc[0] == 10
        assert df["age"].iloc[2] == 30

    def test_cast_bool_to_int_raises(self):
        # bool→int64 direct cast is not supported, raises TypeCastError
        frame = ar.from_pandas(pd.DataFrame({"flag": [True, False, True]}))
        with pytest.raises(ar.TypeCastError, match="Cannot cast column 'flag'"):
            ar.cast_types(frame, {"flag": "int64"})

    def test_cast_int_to_bool(self):
        # 1 → True, 0 → False
        frame = ar.from_pandas(pd.DataFrame({"flag": [1, 0, 1]}))
        result = ar.cast_types(frame, {"flag": "bool"})
        df = ar.to_pandas(result)
        assert result.dtypes["flag"] == "bool"
        assert list(df["flag"]) == [True, False, True]

    def test_cast_same_type_is_noop(self, sample_csv):
        # Casting to the same type should preserve values unchanged
        frame = ar.read_csv(sample_csv)
        result = ar.cast_types(frame, {"age": "int64"})
        df = ar.to_pandas(result)
        assert result.dtypes["age"] == "int64"
        assert list(df["age"]) == [30, 25, 35]

    def test_cast_nonexistent_column_raises(self, sample_csv):
        # Should raise KeyError clearly identifying the missing column
        frame = ar.read_csv(sample_csv)
        with pytest.raises(KeyError, match="nonexistent"):
            ar.cast_types(frame, {"nonexistent": "int64"})

    def test_cast_multiple_columns_at_once(self, sample_csv):
        # Multiple columns in one call should all be cast correctly
        frame = ar.read_csv(sample_csv)
        result = ar.cast_types(frame, {"age": "float64", "name": "string"})
        assert result.dtypes["age"] == "float64"
        assert result.dtypes["name"] == "string"

    def test_cast_string_to_float_unparseable_raises(self):
        # "abc" cannot be parsed as float64, should raise TypeCastError
        frame = ar.from_pandas(pd.DataFrame({"score": ["1.5", "abc"]}))
        with pytest.raises(ar.TypeCastError, match="Cannot cast column 'score'"):
            ar.cast_types(frame, {"score": "float64"})

    def test_cast_string_to_float_unparseable_coerces(self):
        # with errors="coerce", unparseable strings become null
        frame = ar.from_pandas(pd.DataFrame({"score": ["1.5", "abc"]}))
        result = ar.cast_types(frame, {"score": "float64"}, errors="coerce")
        df = ar.to_pandas(result)
        assert result.dtypes["score"] == "float64"
        assert df["score"].iloc[0] == 1.5
        assert pd.isna(df["score"].iloc[1])

    def test_cast_string_to_float_unparseable_ignores_column(self):
        frame = ar.from_pandas(pd.DataFrame({"score": ["1.5", "abc"]}))

        result = ar.cast_types(frame, {"score": "float64"}, errors="ignore")
        df = ar.to_pandas(result)

        assert result.dtypes["score"] == "string"
        assert list(df["score"]) == ["1.5", "abc"]

    def test_cast_string_to_int_unparseable_raises(self):
        # "hello" cannot be parsed as int64, raises TypeCastError by default
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", "hello"]}))
        with pytest.raises(ar.TypeCastError, match="Cannot cast column 'age'"):
            ar.cast_types(frame, {"age": "int64"})

    def test_cast_string_to_int_unparseable_coerces(self):
        # with errors="coerce", unparseable strings become null
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", "hello"]}))
        result = ar.cast_types(frame, {"age": "int64"}, errors="coerce")
        df = ar.to_pandas(result)
        assert result.dtypes["age"] == "int64"
        assert df["age"].iloc[0] == 10
        assert pd.isna(df["age"].iloc[1])

    def test_cast_ignore_casts_valid_columns_and_preserves_invalid_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "age": ["10", "hello"],
                    "score": ["1.5", "2.25"],
                }
            )
        )

        result = ar.cast_types(
            frame,
            {"age": "int64", "score": "float64"},
            errors="ignore",
        )
        df = ar.to_pandas(result)

        assert result.dtypes["age"] == "string"
        assert result.dtypes["score"] == "float64"
        assert list(df["age"]) == ["10", "hello"]
        assert list(df["score"]) == [1.5, 2.25]

    def test_cast_ignore_still_rejects_unknown_target_dtype(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", "20"]}))

        with pytest.raises(ar.TypeCastError, match="Unknown target dtype"):
            ar.cast_types(frame, {"age": "datetime"}, errors="ignore")

    def test_cast_invalid_dtype_string_raises(self):
        # "datetime" is not a supported type, raises TypeCastError
        frame = ar.from_pandas(pd.DataFrame({"age": [1, 2, 3]}))
        with pytest.raises(ar.TypeCastError, match="Unknown target dtype"):
            ar.cast_types(frame, {"age": "datetime"})

    # ------------------------------------------------------------------
    # errors="report" mode
    # ------------------------------------------------------------------

    def test_cast_report_clean_data_returns_empty_failures(self):
        # No bad values → CastReport with empty failures list
        frame = ar.from_pandas(pd.DataFrame({"age": ["1", "2", "3"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        assert isinstance(report, ar.CastReport)
        assert len(report.failures) == 0
        assert not report  # __bool__ is False when no failures

    def test_cast_report_returns_cast_report_type(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["1", "bad"]}))
        result = ar.cast_types(frame, {"age": "int64"}, errors="report")
        assert isinstance(result, ar.CastReport)
        assert isinstance(result.frame, ar.ArFrame)

    def test_cast_report_int_collects_failure(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", "bad", "30"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        assert len(report.failures) == 1
        assert bool(report)  # __bool__ is True when there are failures

    def test_cast_report_failure_fields_are_correct(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", "bad"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        f = report.failures[0]
        assert f.column == "age"
        assert f.row == 1  # 0-based index
        assert f.value == "bad"
        assert f.target_dtype == "int64"

    def test_cast_report_float_collects_failure(self):
        frame = ar.from_pandas(pd.DataFrame({"score": ["1.5", "abc"]}))
        report = ar.cast_types(frame, {"score": "float64"}, errors="report")
        assert len(report.failures) == 1
        f = report.failures[0]
        assert f.column == "score"
        assert f.row == 1
        assert f.value == "abc"
        assert f.target_dtype == "float64"

    def test_cast_report_bool_collects_failure(self):
        frame = ar.from_pandas(pd.DataFrame({"active": ["true", "maybe"]}))
        report = ar.cast_types(frame, {"active": "bool"}, errors="report")
        assert len(report.failures) == 1
        f = report.failures[0]
        assert f.column == "active"
        assert f.value == "maybe"
        assert f.target_dtype == "bool"

    def test_cast_report_null_not_included_in_failures(self):
        # Nulls are preserved as-is — they are not failures
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", None, "30"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        assert len(report.failures) == 0
        df = ar.to_pandas(report.frame)
        assert pd.isna(df["age"].iloc[1])

    def test_cast_report_mixed_valid_and_invalid(self):
        frame = ar.from_pandas(
            pd.DataFrame({"age": ["1", "bad", "3", "also_bad", "5"]})
        )
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        assert len(report.failures) == 2
        assert report.failures[0].row == 1
        assert report.failures[1].row == 3

    def test_cast_report_failure_values_become_null_in_frame(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["10", "bad", "30"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        df = ar.to_pandas(report.frame)
        assert df["age"].iloc[0] == 10
        assert pd.isna(df["age"].iloc[1])  # failure → null
        assert df["age"].iloc[2] == 30

    def test_cast_report_all_bad_values_no_raise(self):
        # report mode must never raise, even when every value fails
        frame = ar.from_pandas(pd.DataFrame({"age": ["a", "b", "c"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        assert len(report.failures) == 3
        df = ar.to_pandas(report.frame)
        assert df["age"].isna().all()

    def test_cast_report_frame_dtype_matches_target(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["1", "bad"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        assert report.frame.dtypes["age"] == "int64"

    def test_cast_report_multi_column_collects_across_columns(self):
        frame = ar.from_pandas(
            pd.DataFrame({"age": ["1", "bad"], "score": ["1.5", "abc"]})
        )
        report = ar.cast_types(
            frame, {"age": "int64", "score": "float64"}, errors="report"
        )
        columns = [f.column for f in report.failures]
        assert "age" in columns
        assert "score" in columns

    def test_cast_report_failures_ordered_by_row(self):
        frame = ar.from_pandas(pd.DataFrame({"age": ["bad", "1", "also_bad"]}))
        report = ar.cast_types(frame, {"age": "int64"}, errors="report")
        rows = [f.row for f in report.failures]
        assert rows == sorted(rows)

    def test_cast_report_multi_column_failures_ordered_by_row(self):
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "age": ["1", "bad"],
                    "score": ["bad", "2.5"],
                }
            )
        )
        report = ar.cast_types(
            frame, {"age": "int64", "score": "float64"}, errors="report"
        )
        assert [f.row for f in report.failures] == [0, 1]


class TestCleanAPI:
    def test_clean_defaults(self, csv_with_whitespace):
        frame = ar.read_csv(csv_with_whitespace)
        result = ar.clean(frame)
        df = ar.to_pandas(result)

        # strip_whitespace is True by default
        assert df["name"].iloc[0] == "Alice"
        assert df["city"].iloc[1] == "London"

        # drop_nulls and drop_duplicates are False by default
        assert len(frame) == len(result)

    def test_clean_all(self, csv_with_nulls):

        frame = ar.read_csv(csv_with_nulls)

        result = ar.clean(
            frame,
            strip_whitespace=False,
            drop_nulls=True,
        )

        assert len(result) < len(frame)


class TestWinsorizeOutliers:
    def test_winsorize_actual_values_capped(self):
        """Verify values are actually capped, not just type-checked."""
        import pandas as pd

        df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 1000.0]})
        frame = ar.from_pandas(df)
        clean = ar.winsorize_outliers(frame, lower=0.05, upper=0.95)
        result_df = ar.to_pandas(clean)
        assert result_df["price"].max() < 1000.0

    def test_winsorize_identical_values(self):
        """Frame where all values are identical should not crash."""
        import pandas as pd

        df = pd.DataFrame({"score": [5.0, 5.0, 5.0, 5.0]})
        frame = ar.from_pandas(df)
        clean = ar.winsorize_outliers(frame, lower=0.05, upper=0.95)
        assert isinstance(clean, ar.ArFrame)

    def test_winsorize_single_row(self):
        """Single row frame should not crash."""
        import pandas as pd

        df = pd.DataFrame({"score": [42.0]})
        frame = ar.from_pandas(df)
        clean = ar.winsorize_outliers(frame, lower=0.05, upper=0.95)
        assert isinstance(clean, ar.ArFrame)

    def test_winsorize_unknown_subset_column_raises(self):
        """Unknown column in subset should raise ValueError."""
        import pandas as pd

        df = pd.DataFrame({"age": [25, 30, 35]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.winsorize_outliers(frame, subset=["nonexistent"])

    def test_winsorize_caps_upper_outlier(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        clean = ar.winsorize_outliers(frame, lower=0.05, upper=0.95)
        assert isinstance(clean, ar.ArFrame)

    def test_winsorize_returns_same_row_count(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        clean = ar.winsorize_outliers(frame, lower=0.05, upper=0.95)
        assert len(clean) == len(frame)

    def test_winsorize_subset_only(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        clean = ar.winsorize_outliers(frame, lower=0.05, upper=0.95, subset=["age"])
        assert isinstance(clean, ar.ArFrame)

    def test_winsorize_skips_string_columns(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        clean = ar.winsorize_outliers(frame, lower=0.05, upper=0.95)
        assert isinstance(clean, ar.ArFrame)

    def test_winsorize_in_pipeline(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        clean = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("winsorize_outliers", {"lower": 0.05, "upper": 0.95}),
            ],
        )
        assert isinstance(clean, ar.ArFrame)

    def test_winsorize_invalid_lower_greater_than_upper(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(ValueError):
            ar.winsorize_outliers(frame, lower=0.9, upper=0.1)

    def test_winsorize_invalid_lower_equals_upper(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(ValueError):
            ar.winsorize_outliers(frame, lower=0.5, upper=0.5)

    def test_winsorize_invalid_out_of_range(self, sample_csv):
        frame = ar.read_csv(sample_csv)
        with pytest.raises(ValueError):
            ar.winsorize_outliers(frame, lower=-0.1, upper=1.5)
class TestFilterRows:
    def test_filter_rows_missing_column_raises_clear_error(self):
        df = pd.DataFrame({"age": [20, 30]})

        with pytest.raises(ValueError, match="Unknown column: missing"):
            ar.filter_rows(df, "missing", ">", 10)

    def test_filter_rows_missing_column_raises_clear_error_for_arframe(self):
        frame = ar.from_pandas(pd.DataFrame({"age": [20, 30]}))

        with pytest.raises(ValueError, match="Unknown column: missing"):
            ar.filter_rows(frame, "missing", ">", 10)

    def test_filter_rows_valid_column_still_works(self):
        df = pd.DataFrame({"age": [20, 30]})

        result = ar.filter_rows(df, "age", ">", 20)

        assert len(result) == 1
        assert result.iloc[0]["age"] == 30

    def test_filter_rows_with_missing_values_does_not_crash(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({"age": [20, 30, np.nan, pd.NA, None]})

        result = ar.filter_rows(df, "age", ">", 25)

        assert len(result) == 1
        assert result.iloc[0]["age"] == 30

    def test_filter_rows_arframe_resets_row_positions(self):
        frame = ar.from_pandas(pd.DataFrame({"age": [10, 30, 40]}))

        result = ar.filter_rows(frame, "age", ">", 20)
        df = ar.to_pandas(result)

        assert list(df.index) == [0, 1]
        assert list(df["age"]) == [30, 40]

    def test_filter_rows_invalid_comparison_raises_column_aware_type_error(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"]})

        with pytest.raises(
            TypeError, match="filter_rows: cannot compare column 'name'"
        ):
            ar.filter_rows(df, "name", ">", 1)


class TestMappingValidation:
    def test_rename_columns_rejects_invalid_mapping_value_type(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="mapping values must be non-empty strings"):
            ar.rename_columns(frame, {"name": 123})

    def test_replace_values_rejects_missing_column(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(KeyError, match="Column 'missing' not found"):
            ar.replace_values(frame, {"Alice": "Alicia"}, column="missing")

    def test_replace_values_rejects_non_mapping_input(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="mapping must be a dict-like mapping"):
            ar.replace_values(frame, [("Alice", "Alicia")])

    def test_replace_values_rejects_empty_mapping(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(ValueError, match="mapping must not be empty"):
            ar.replace_values(frame, {})

    def test_rename_columns_rejects_non_mapping_input(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="mapping must be a mapping"):
            ar.rename_columns(frame, [("name", "full_name")])

    def test_cast_types_rejects_non_mapping_input(self, sample_csv):
        frame = ar.read_csv(sample_csv)

        with pytest.raises(TypeError, match="mapping must be a mapping"):
            ar.cast_types(frame, [("age", "string")])


class TestReplaceValues:
    def test_replace_values_null_key_replaces_existing_nulls_in_target_column(self):
        import numpy as np

        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "name": ["Alice", None, pd.NA],
                    "city": [None, "Paris", None],
                }
            )
        )

        result = ar.replace_values(frame, {np.nan: "Unknown"}, column="name")
        df = ar.to_pandas(result)

        assert list(df["name"]) == ["Alice", "Unknown", "Unknown"]
        assert pd.isna(df.loc[0, "city"])
        assert df.loc[1, "city"] == "Paris"
        assert pd.isna(df.loc[2, "city"])

    def test_replace_values_null_replacement_creates_real_nulls(self):
        frame = ar.from_pandas(
            pd.DataFrame({"status": ["active", "inactive", "active"]})
        )

        result = ar.replace_values(frame, {"inactive": None})
        df = ar.to_pandas(result)

        assert list(df["status"].iloc[[0, 2]]) == ["active", "active"]
        assert pd.isna(df.loc[1, "status"])

    def test_replace_values_supports_pd_na_key_and_value(self):
        frame = ar.from_pandas(
            pd.DataFrame({"score": [1, None, 3], "flag": ["ok", "missing", "ok"]})
        )

        result = ar.replace_values(frame, {pd.NA: 0, "missing": pd.NA})
        df = ar.to_pandas(result)

        assert list(df["score"]) == [1, 0, 3]
        assert df.loc[0, "flag"] == "ok"
        assert pd.isna(df.loc[1, "flag"])
        assert df.loc[2, "flag"] == "ok"

    def test_replace_values_tuple_mapping_key_does_not_crash(self):
        frame = ar.from_pandas(pd.DataFrame({"col": ["A", "B", "C"]}))

        result = ar.replace_values(
            frame,
            {("A", "B"): "X"},
            column="col",
        )

        df = ar.to_pandas(result)

        assert list(df["col"]) == ["A", "B", "C"]

    def test_replace_values_mixed_tuple_and_null_keys(self):
        frame = ar.from_pandas(pd.DataFrame({"col": ["A", np.nan, "C"]}))

        result = ar.replace_values(
            frame,
            {
                ("A", "B"): "X",
                np.nan: "missing",
            },
            column="col",
        )

        df = ar.to_pandas(result)

        assert list(df["col"]) == ["A", "missing", "C"]


class TestRoundNumericColumns:
    def test_round_all_numeric(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.123, 2.456], "b": [3.789, 4.0]})
        frame = ar.from_pandas(df)
        result = ar.round_numeric_columns(frame, decimals=1)
        result_df = ar.to_pandas(result)
        assert list(result_df["a"]) == [1.1, 2.5]
        assert list(result_df["b"]) == [3.8, 4.0]

    def test_round_subset(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.123, 2.456], "b": [3.789, 4.0]})
        frame = ar.from_pandas(df)
        result = ar.round_numeric_columns(frame, subset=["a"], decimals=1)
        result_df = ar.to_pandas(result)
        assert list(result_df["a"]) == [1.1, 2.5]
        assert list(result_df["b"]) == [3.789, 4.0]

    def test_round_mixed_types(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.123, 2.456], "c": ["str1", "str2"]})
        frame = ar.from_pandas(df)
        result = ar.round_numeric_columns(frame, decimals=1)
        result_df = ar.to_pandas(result)
        assert list(result_df["a"]) == [1.1, 2.5]
        assert list(result_df["c"]) == ["str1", "str2"]

    def test_missing_column(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.123]})
        frame = ar.from_pandas(df)
        with pytest.raises(IndexError, match="Column not found"):
            ar.round_numeric_columns(frame, subset=["missing_col"])

    def test_with_nulls(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({"a": [1.123, np.nan, 2.456]})
        frame = ar.from_pandas(df)
        result = ar.round_numeric_columns(frame, decimals=1)
        result_df = ar.to_pandas(result)
        assert result_df["a"].isna().iloc[1]
        assert result_df["a"].iloc[0] == 1.1
        assert result_df["a"].iloc[2] == 2.5

    def test_invalid_subset_type(self):
        import pandas as pd
        import pytest

        df = pd.DataFrame({"a": [1.123]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="subset must be a list"):
            ar.round_numeric_columns(frame, subset="a")

    def test_invalid_decimals_type(self):
        import pandas as pd
        import pytest

        df = pd.DataFrame({"a": [1.123]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="decimals must be an integer"):
            ar.round_numeric_columns(frame, decimals="2")

    def test_decimals_rejects_bool(self):
        import pandas as pd
        import pytest

        df = pd.DataFrame({"a": [1.123]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="decimals must be an integer"):
            ar.round_numeric_columns(frame, decimals=True)

    def test_round_subset_with_non_numeric(self):
        import pandas as pd

        df = pd.DataFrame({"name": ["john"], "score": [98.765]})
        frame = ar.from_pandas(df)
        result = ar.round_numeric_columns(frame, subset=["name", "score"], decimals=1)
        result_df = ar.to_pandas(result)

        assert list(result_df["name"]) == ["john"]
        assert list(result_df["score"]) == [98.8]


class TestSafeDivideColumns:
    def test_normal_division(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("revenue,cost\n100,50\n200,100\n300,150\n")
        frame = ar.read_csv(path)
        result = ar.safe_divide_columns(
            frame, numerator="revenue", denominator="cost", output_column="ratio"
        )
        df = ar.to_pandas(result)
        assert df["ratio"].iloc[0] == 2.0
        assert df["ratio"].iloc[1] == 2.0
        assert df["ratio"].iloc[2] == 2.0

    def test_division_by_zero(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("revenue,cost\n100,0\n200,100\n300,0\n")
        frame = ar.read_csv(path)
        result = ar.safe_divide_columns(
            frame, numerator="revenue", denominator="cost", output_column="ratio"
        )
        df = ar.to_pandas(result)
        assert df["ratio"].iloc[0] == 0.0
        assert df["ratio"].iloc[2] == 0.0

    def test_null_inputs(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("revenue,cost\n100,\n200,100\n300,\n")
        frame = ar.read_csv(path)
        result = ar.safe_divide_columns(
            frame, numerator="revenue", denominator="cost", output_column="ratio"
        )
        df = ar.to_pandas(result)
        assert df["ratio"].iloc[0] == 0.0
        assert df["ratio"].iloc[2] == 0.0

    def test_missing_numerator_column(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("revenue,cost\n100,50\n")
        frame = ar.read_csv(path)
        with pytest.raises(ValueError, match="Numerator column"):
            ar.safe_divide_columns(
                frame,
                numerator="nonexistent",
                denominator="cost",
                output_column="ratio",
            )

    def test_missing_denominator_column(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("revenue,cost\n100,50\n")
        frame = ar.read_csv(path)
        with pytest.raises(ValueError, match="Denominator column"):
            ar.safe_divide_columns(
                frame,
                numerator="revenue",
                denominator="nonexistent",
                output_column="ratio",
            )

    def test_output_column_already_exists(self, tmp_path):
        import warnings

        path = tmp_path / "data.csv"
        path.write_text("revenue,cost,ratio\n100,50,99\n200,100,99\n")
        frame = ar.read_csv(path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ar.safe_divide_columns(
                frame, numerator="revenue", denominator="cost", output_column="ratio"
            )
            assert len(w) == 1
            assert "already exists" in str(w[0].message)
        df = ar.to_pandas(result)
        assert df["ratio"].iloc[0] == 2.0

# for the issue (#345)
class TestRemoveSpecialChars:
    def test_all_string_columns(self):
        df = pd.DataFrame({
            "name": ["An@shu#", "Jo!hn"],
            "city": ["Kol!kata", "Del#hi"],
            "age": [20, 25]
        })

        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("remove_special_chars",)])
        cleaned = ar.to_pandas(result)

        assert cleaned["name"].tolist() == ["Anshu", "John"]
        assert cleaned["city"].tolist() == ["Kolkata", "Delhi"]
        assert cleaned["age"].tolist() == [20, 25]


    def test_subset_columns(self):
        df = pd.DataFrame({
            "name": ["An@shu#"],
            "city": ["Kol!kata"]
        })

        frame = ar.from_pandas(df)
        result = ar.pipeline(
            frame,
            [("remove_special_chars", {"columns": ["name"]})]
        )
        cleaned = ar.to_pandas(result)

        assert cleaned["name"][0] == "Anshu"
        assert cleaned["city"][0] == "Kol!kata"


    def test_no_special_chars(self):
        df = pd.DataFrame({
            "name": ["Anshu"]
        })

        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("remove_special_chars",)])
        cleaned = ar.to_pandas(result)

        assert cleaned["name"][0] == "Anshu"


    def test_non_string_columns_ignored(self):
        df = pd.DataFrame({
            "age": [10, 20]
        })

        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("remove_special_chars",)])
        cleaned = ar.to_pandas(result)

        assert cleaned["age"].tolist() == [10, 20]


    def test_invalid_column_raises(self):
        df = pd.DataFrame({
            "name": ["Anshu"]
        })

        frame = ar.from_pandas(df)

        with pytest.raises(ValueError):
            ar.pipeline(
                frame,
                [("remove_special_chars", {"columns": ["wrong"]})]
            )