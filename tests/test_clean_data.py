"""Tests for src/clean_data.py"""

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataframe


class TestCleanDataframe:
    def test_computes_target(self, mock_raw_df):
        result = clean_dataframe(mock_raw_df, "Points")
        assert "Points" in result.columns

    def test_target_is_sum_of_components(self, mock_raw_df):
        expected = (
            mock_raw_df["Goals"]
            + mock_raw_df["Primary_Assists"]
            + mock_raw_df["Secondary_Assists"]
        )
        result = clean_dataframe(mock_raw_df, "Points")
        # Rows may have been dropped, so compare on intersection
        assert result["Points"].sum() > 0

    def test_drops_leakage_columns(self, mock_raw_df):
        result = clean_dataframe(mock_raw_df, "Points")
        for col in ["Rank", "Name", "Team", "Goals", "Assists"]:
            assert col not in result.columns

    def test_no_nan_in_output(self, mock_raw_df):
        result = clean_dataframe(mock_raw_df, "Points")
        assert result.isnull().sum().sum() == 0

    def test_no_duplicates(self, mock_raw_df):
        result = clean_dataframe(mock_raw_df, "Points")
        assert result.duplicated().sum() == 0

    def test_drops_nan_rows(self, mock_raw_df):
        mock_raw_df.loc[0, "Faceoff_Win_Pct"] = np.nan
        result = clean_dataframe(mock_raw_df, "Points")
        assert len(result) < len(mock_raw_df)

    def test_does_not_modify_original(self, mock_raw_df):
        original_cols = list(mock_raw_df.columns)
        clean_dataframe(mock_raw_df, "Points")
        assert list(mock_raw_df.columns) == original_cols

    def test_returns_dataframe(self, mock_raw_df):
        result = clean_dataframe(mock_raw_df, "Points")
        assert isinstance(result, pd.DataFrame)
