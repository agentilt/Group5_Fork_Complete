"""Tests for src/validate.py"""

import pandas as pd
import pytest

from src.utils import DataValidationError
from src.validate import (
    validate_columns,
    validate_dataframe,
    validate_no_leakage,
    validate_no_nulls,
    validate_not_empty,
    validate_numeric_columns,
)


class TestValidateNotEmpty:
    """Tests for validate_not_empty."""

    def test_empty_raises(self):
        with pytest.raises(DataValidationError, match="empty"):
            validate_not_empty(pd.DataFrame())

    def test_non_empty_passes(self):
        df = pd.DataFrame({"a": [1]})
        validate_not_empty(df)  # Should not raise


class TestValidateColumns:
    """Tests for validate_columns."""

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(DataValidationError, match="Missing"):
            validate_columns(df, ["a", "b"])

    def test_all_present_passes(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        validate_columns(df, ["a", "b"])  # Should not raise

    def test_empty_required_list_passes(self):
        df = pd.DataFrame({"a": [1]})
        validate_columns(df, [])  # Should not raise


class TestValidateNoNulls:
    """Tests for validate_no_nulls."""

    def test_nulls_raise(self):
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        with pytest.raises(DataValidationError, match="null"):
            validate_no_nulls(df)

    def test_no_nulls_passes(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        validate_no_nulls(df)  # Should not raise


class TestValidateNumericColumns:
    """Tests for validate_numeric_columns."""

    def test_non_numeric_raises(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        with pytest.raises(DataValidationError, match="non-numeric"):
            validate_numeric_columns(df, ["a"])

    def test_numeric_passes(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        validate_numeric_columns(df, ["a", "b"])  # Should not raise

    def test_skips_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        validate_numeric_columns(df, ["z"])  # Column not present, skip


class TestValidateNoLeakage:
    """Tests for validate_no_leakage."""

    def test_leakage_detected_raises(self):
        df = pd.DataFrame({"Goals": [10], "Points": [20]})
        with pytest.raises(DataValidationError, match="leakage"):
            validate_no_leakage(df, ["Goals", "Name"])

    def test_no_leakage_passes(self):
        df = pd.DataFrame({"Points": [20], "Shot_Attempts": [100]})
        validate_no_leakage(df, ["Goals", "Name"])  # Should not raise


class TestValidateDataframe:
    """Tests for the full validate_dataframe function."""

    def test_valid_dataframe_passes(self, mock_clean_df, sample_config):
        assert validate_dataframe(mock_clean_df, sample_config) is True

    def test_empty_dataframe_raises(self, sample_config):
        with pytest.raises(DataValidationError):
            validate_dataframe(pd.DataFrame(), sample_config)

    def test_missing_target_raises(self, mock_clean_df, sample_config):
        df = mock_clean_df.drop(columns=["Points"])
        with pytest.raises(DataValidationError):
            validate_dataframe(df, sample_config)

    def test_detects_nan(self, mock_clean_df, sample_config):
        mock_clean_df.loc[0, "Takeaways"] = None
        with pytest.raises(DataValidationError, match="null"):
            validate_dataframe(mock_clean_df, sample_config)
