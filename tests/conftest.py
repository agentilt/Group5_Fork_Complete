"""Shared test fixtures for the NHL pipeline test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_raw_df():
    """Mock raw NHL DataFrame with all columns needed by the pipeline."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Rank": range(1, n + 1),
        "Team": np.random.choice(["Team A", "Team B"], size=n),
        "Name": [f"Player_{i}" for i in range(1, n + 1)],
        "Pos": np.random.choice(["C", "D", "L", "R"], size=n),
        "Games_Played": np.random.randint(50, 82, n),
        "Icetime_Minutes": np.random.uniform(800, 1800, n).round(1),
        "Goals": np.random.randint(0, 40, n),
        "Assists": np.random.randint(0, 50, n),
        "Primary_Assists": np.random.randint(0, 30, n),
        "Secondary_Assists": np.random.randint(0, 20, n),
        "Faceoff_Win_Pct": np.random.uniform(30, 65, n).round(1),
        "Takeaways": np.random.randint(5, 60, n),
        "Giveaways": np.random.randint(10, 80, n),
        "Shot_Attempts": np.random.randint(80, 400, n),
        "Shooting_Pct_On_Unblocked": np.random.uniform(3, 18, n).round(1),
        "PIM_Drawn": np.random.randint(5, 60, n),
        "Pct_Shift_Starts_Offensive_Zone": np.random.uniform(8, 25, n).round(1),
        "On_Ice_Corsi_Pct": np.random.uniform(40, 60, n).round(1),
    })


@pytest.fixture
def mock_clean_df(mock_raw_df):
    """Mock cleaned DataFrame (target computed, leakage dropped)."""
    from src.clean_data import clean_dataframe
    return clean_dataframe(mock_raw_df, "Points")


@pytest.fixture
def tmp_csv(tmp_path, mock_raw_df):
    """Save mock raw DataFrame to a temporary CSV and return the path."""
    csv_path = tmp_path / "test_data.csv"
    mock_raw_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def settings():
    """Return the SETTINGS dict matching main.py."""
    return {
        "is_example_config": False,
        "target_column": "Points",
        "problem_type": "regression",
        "test_size": 0.2,
        "random_state": 42,
        "features": {
            "quantile_bin": ["Icetime_Minutes", "Shot_Attempts"],
            "categorical_onehot": ["Pos"],
            "numeric_passthrough": [
                "Faceoff_Win_Pct", "Takeaways", "Giveaways",
                "Shooting_Pct_On_Unblocked", "PIM_Drawn",
                "Pct_Shift_Starts_Offensive_Zone", "On_Ice_Corsi_Pct",
            ],
            "n_bins": 3,
        },
    }
