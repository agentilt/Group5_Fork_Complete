"""Tests for src/validate.py"""

import pandas as pd
import pytest

from src.validate import validate_dataframe


class TestValidateDataframe:
    def test_valid_passes(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validate_dataframe(df, ["a", "b"]) is True

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(pd.DataFrame(), ["a"])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing"):
            validate_dataframe(df, ["a", "b"])

    def test_nulls_raise(self):
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        with pytest.raises(ValueError, match="nulls"):
            validate_dataframe(df, ["a"])

    def test_empty_required_list_passes(self):
        df = pd.DataFrame({"a": [1]})
        assert validate_dataframe(df, []) is True

    def test_works_with_clean_data(self, mock_clean_df):
        assert validate_dataframe(mock_clean_df, ["Points"]) is True
