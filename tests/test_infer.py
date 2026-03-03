"""Tests for src/infer.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.infer import (
    identify_undervalued_players,
    predict,
    prepare_inference_data,
)


@pytest.fixture
def mock_inference_setup(sample_config):
    """Create a fitted scaler and model matching the expected features."""
    np.random.seed(42)
    n = 50

    # Expected features after encoding (no Pos, has Pos_D, Pos_L, Pos_R)
    feature_cols = [
        "Faceoff_Win_Pct", "Takeaways", "Giveaways", "Shot_Attempts",
        "Shooting_Pct_On_Unblocked", "PIM_Drawn", "Icetime_Minutes",
        "Pct_Shift_Starts_Offensive_Zone", "On_Ice_Corsi_Pct",
        "Pos_D", "Pos_L", "Pos_R",
    ]
    X_train = pd.DataFrame(
        np.random.randn(n, len(feature_cols)), columns=feature_cols,
    )

    scaler = StandardScaler().fit(X_train)
    model = LinearRegression().fit(X_train, np.random.randn(n))

    return model, scaler


@pytest.fixture
def mock_inference_df():
    """Raw data for inference (same structure as training input)."""
    np.random.seed(99)
    n = 10
    return pd.DataFrame({
        "Name": [f"Player_{i}" for i in range(n)],
        "Pos": np.random.choice(["C", "D", "L", "R"], size=n),
        "Goals": np.random.randint(0, 30, n),
        "Primary_Assists": np.random.randint(0, 20, n),
        "Secondary_Assists": np.random.randint(0, 15, n),
        "Faceoff_Win_Pct": np.random.uniform(30, 60, n).round(1),
        "Takeaways": np.random.randint(5, 50, n),
        "Giveaways": np.random.randint(10, 70, n),
        "Shot_Attempts": np.random.randint(80, 300, n),
        "Shooting_Pct_On_Unblocked": np.random.uniform(3, 18, n).round(1),
        "PIM_Drawn": np.random.randint(5, 50, n),
        "Icetime_Minutes": np.random.uniform(800, 1800, n).round(1),
        "Pct_Shift_Starts_Offensive_Zone": np.random.uniform(8, 25, n).round(1),
        "On_Ice_Corsi_Pct": np.random.uniform(40, 60, n).round(1),
    })


class TestPrepareInferenceData:
    """Tests for prepare_inference_data."""

    def test_returns_scaled_data(
        self, mock_inference_df, sample_config, mock_inference_setup
    ):
        _, scaler = mock_inference_setup
        X_scaled, names, y_actual = prepare_inference_data(
            mock_inference_df, sample_config, scaler,
        )
        assert isinstance(X_scaled, pd.DataFrame)
        assert len(X_scaled) == 10

    def test_preserves_names(
        self, mock_inference_df, sample_config, mock_inference_setup
    ):
        _, scaler = mock_inference_setup
        _, names, _ = prepare_inference_data(
            mock_inference_df, sample_config, scaler,
        )
        assert names is not None
        assert len(names) == 10

    def test_computes_target_if_components_present(
        self, mock_inference_df, sample_config, mock_inference_setup
    ):
        _, scaler = mock_inference_setup
        _, _, y_actual = prepare_inference_data(
            mock_inference_df, sample_config, scaler,
        )
        assert y_actual is not None

    def test_missing_column_raises(self, sample_config, mock_inference_setup):
        _, scaler = mock_inference_setup
        df = pd.DataFrame({"Name": ["A"], "Pos": ["C"]})
        with pytest.raises(ValueError, match="missing columns"):
            prepare_inference_data(df, sample_config, scaler)


class TestPredict:
    """Tests for the predict function."""

    def test_returns_series(self, mock_inference_setup):
        model, _ = mock_inference_setup
        X = pd.DataFrame(np.random.randn(5, 12), columns=[
            "Faceoff_Win_Pct", "Takeaways", "Giveaways", "Shot_Attempts",
            "Shooting_Pct_On_Unblocked", "PIM_Drawn", "Icetime_Minutes",
            "Pct_Shift_Starts_Offensive_Zone", "On_Ice_Corsi_Pct",
            "Pos_D", "Pos_L", "Pos_R",
        ])
        result = predict(model, X)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_prediction_name(self, mock_inference_setup):
        model, _ = mock_inference_setup
        X = pd.DataFrame(np.random.randn(3, 12), columns=[
            "Faceoff_Win_Pct", "Takeaways", "Giveaways", "Shot_Attempts",
            "Shooting_Pct_On_Unblocked", "PIM_Drawn", "Icetime_Minutes",
            "Pct_Shift_Starts_Offensive_Zone", "On_Ice_Corsi_Pct",
            "Pos_D", "Pos_L", "Pos_R",
        ])
        result = predict(model, X)
        assert result.name == "Predicted_Points"


class TestIdentifyUndervaluedPlayers:
    """Tests for identify_undervalued_players."""

    def test_finds_undervalued(self):
        names = pd.Series(["Alice", "Bob", "Charlie"])
        actual = pd.Series([20, 30, 40])
        predicted = pd.Series([30, 31, 42])

        result = identify_undervalued_players(
            names, actual, predicted, threshold=5.0
        )
        assert len(result) == 1
        assert result.iloc[0]["Name"] == "Alice"

    def test_empty_when_no_undervalued(self):
        names = pd.Series(["Alice", "Bob"])
        actual = pd.Series([30, 40])
        predicted = pd.Series([28, 39])

        result = identify_undervalued_players(
            names, actual, predicted, threshold=5.0
        )
        assert len(result) == 0

    def test_sorted_by_difference(self):
        names = pd.Series(["A", "B", "C"])
        actual = pd.Series([10, 10, 10])
        predicted = pd.Series([20, 25, 18])

        result = identify_undervalued_players(
            names, actual, predicted, threshold=5.0
        )
        assert result.iloc[0]["Name"] == "B"
        assert result.iloc[1]["Name"] == "A"
