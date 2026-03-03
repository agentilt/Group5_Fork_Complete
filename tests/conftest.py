"""
Shared test fixtures for the NHL pipeline test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config():
    """Return a minimal valid pipeline configuration."""
    return {
        "data": {
            "raw": "data/raw/nhl_player_stats.csv",
            "processed": "data/processed/clean.csv",
            "inference": "data/inference/",
        },
        "model": {
            "output_dir": "models/",
            "ols_path": "models/ols_model.pkl",
            "lasso_path": "models/lasso_model.pkl",
            "scaler_path": "models/scaler.pkl",
        },
        "train": {
            "test_size": 0.2,
            "seed": 42,
            "lasso_cv_folds": 5,
        },
        "features": {
            "target": "Points",
            "target_components": [
                "Goals", "Primary_Assists", "Secondary_Assists",
            ],
            "explanatory": [
                "Faceoff_Win_Pct", "Takeaways", "Giveaways",
                "Shot_Attempts", "Shooting_Pct_On_Unblocked", "PIM_Drawn",
            ],
            "controls": [
                "Icetime_Minutes", "Pos",
                "Pct_Shift_Starts_Offensive_Zone", "On_Ice_Corsi_Pct",
            ],
            "leakage_columns": [
                "Rank", "Name", "Team", "Goals", "Assists",
                "Primary_Assists", "Secondary_Assists",
            ],
        },
        "reports": {"output_dir": "reports/"},
        "logging": {
            "log_file": "logs/test_pipeline.log",
            "level": "DEBUG",
        },
    }


@pytest.fixture
def mock_raw_df():
    """Create a mock raw NHL DataFrame mimicking the real dataset."""
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
        "Points": np.random.randint(5, 100, n),
    })


@pytest.fixture
def mock_clean_df():
    """Create a mock cleaned and encoded DataFrame (after clean_dataframe)."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Faceoff_Win_Pct": np.random.uniform(30, 65, n).round(1),
        "Takeaways": np.random.randint(5, 60, n),
        "Giveaways": np.random.randint(10, 80, n),
        "Shot_Attempts": np.random.randint(80, 400, n),
        "Shooting_Pct_On_Unblocked": np.random.uniform(3, 18, n).round(1),
        "PIM_Drawn": np.random.randint(5, 60, n),
        "Icetime_Minutes": np.random.uniform(800, 1800, n).round(1),
        "Pct_Shift_Starts_Offensive_Zone": np.random.uniform(8, 25, n).round(1),
        "On_Ice_Corsi_Pct": np.random.uniform(40, 60, n).round(1),
        "Points": np.random.randint(5, 100, n),
        "Pos_D": np.random.choice([0, 1], size=n),
        "Pos_L": np.random.choice([0, 1], size=n),
        "Pos_R": np.random.choice([0, 1], size=n),
    })


@pytest.fixture
def tmp_csv(tmp_path, mock_raw_df):
    """Save a mock raw DataFrame to a temporary CSV and return the path."""
    csv_path = tmp_path / "test_data.csv"
    mock_raw_df.to_csv(csv_path, index=False)
    return csv_path
