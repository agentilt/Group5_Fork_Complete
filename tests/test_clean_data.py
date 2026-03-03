"""Tests for src/clean_data.py"""

import numpy as np
import pandas as pd
import pytest

from src.clean_data import (
    clean_dataframe,
    compute_target,
    encode_categoricals,
    select_features,
)
from src.utils import DataValidationError


class TestComputeTarget:
    """Tests for the compute_target function."""

    def test_target_is_sum_of_components(self):
        df = pd.DataFrame({
            "Goals": [10, 20],
            "Primary_Assists": [5, 10],
            "Secondary_Assists": [3, 7],
        })
        result = compute_target(df, "Points", [
            "Goals", "Primary_Assists", "Secondary_Assists",
        ])
        assert list(result["Points"]) == [18, 37]

    def test_missing_component_raises(self):
        df = pd.DataFrame({"Goals": [10], "Primary_Assists": [5]})
        with pytest.raises(DataValidationError, match="missing columns"):
            compute_target(df, "Points", [
                "Goals", "Primary_Assists", "Secondary_Assists",
            ])

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            "Goals": [10], "Primary_Assists": [5],
            "Secondary_Assists": [3],
        })
        original_cols = list(df.columns)
        compute_target(df, "Points", [
            "Goals", "Primary_Assists", "Secondary_Assists",
        ])
        assert list(df.columns) == original_cols

    def test_handles_zero_values(self):
        df = pd.DataFrame({
            "Goals": [0], "Primary_Assists": [0],
            "Secondary_Assists": [0],
        })
        result = compute_target(df, "Points", [
            "Goals", "Primary_Assists", "Secondary_Assists",
        ])
        assert result["Points"].iloc[0] == 0


class TestSelectFeatures:
    """Tests for the select_features function."""

    def test_selects_correct_columns(self, mock_raw_df, sample_config):
        result = select_features(mock_raw_df, sample_config)
        expected = (
            sample_config["features"]["explanatory"]
            + sample_config["features"]["controls"]
            + [sample_config["features"]["target"]]
        )
        assert list(result.columns) == expected

    def test_missing_feature_raises(self, sample_config):
        df = pd.DataFrame({"Faceoff_Win_Pct": [50.0]})
        with pytest.raises(DataValidationError, match="missing columns"):
            select_features(df, sample_config)

    def test_drops_extra_columns(self, mock_raw_df, sample_config):
        result = select_features(mock_raw_df, sample_config)
        # Rank, Name, Team should not be in the output
        assert "Rank" not in result.columns
        assert "Name" not in result.columns


class TestEncodeCategoricals:
    """Tests for the encode_categoricals function."""

    def test_creates_dummy_columns(self):
        df = pd.DataFrame({
            "Pos": ["C", "D", "L", "R"],
            "Value": [1, 2, 3, 4],
        })
        result = encode_categoricals(df, "Pos", "C")
        assert "Pos_D" in result.columns
        assert "Pos_L" in result.columns
        assert "Pos_R" in result.columns
        assert "Pos" not in result.columns

    def test_baseline_dropped(self):
        df = pd.DataFrame({"Pos": ["C", "D", "L", "R"]})
        result = encode_categoricals(df, "Pos", "C")
        # C is baseline, so no Pos_C column
        assert "Pos_C" not in result.columns

    def test_skips_missing_column(self):
        df = pd.DataFrame({"Value": [1, 2]})
        result = encode_categoricals(df, "Pos", "C")
        pd.testing.assert_frame_equal(df, result)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"Pos": ["C", "D"], "Value": [1, 2]})
        encode_categoricals(df, "Pos", "C")
        assert "Pos" in df.columns


class TestCleanDataframe:
    """Tests for the full clean_dataframe pipeline."""

    def test_end_to_end(self, mock_raw_df, sample_config):
        result = clean_dataframe(mock_raw_df, sample_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "Points" in result.columns

    def test_no_nan_in_output(self, mock_raw_df, sample_config):
        result = clean_dataframe(mock_raw_df, sample_config)
        assert result.isnull().sum().sum() == 0

    def test_no_duplicates_in_output(self, mock_raw_df, sample_config):
        result = clean_dataframe(mock_raw_df, sample_config)
        assert result.duplicated().sum() == 0

    def test_pos_encoded(self, mock_raw_df, sample_config):
        result = clean_dataframe(mock_raw_df, sample_config)
        assert "Pos" not in result.columns
        assert "Pos_D" in result.columns

    def test_drops_nan_rows(self, mock_raw_df, sample_config):
        # Inject a NaN into a feature column
        mock_raw_df.loc[0, "Faceoff_Win_Pct"] = np.nan
        result = clean_dataframe(mock_raw_df, sample_config)
        assert len(result) < len(mock_raw_df)

    def test_leakage_columns_removed(self, mock_raw_df, sample_config):
        result = clean_dataframe(mock_raw_df, sample_config)
        for col in ["Rank", "Name", "Team", "Assists"]:
            assert col not in result.columns
