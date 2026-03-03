"""Tests for src/main.py"""

import pytest

from src.main import run_training_pipeline
from src.utils import load_config


class TestRunTrainingPipeline:
    """Integration tests for the full training pipeline."""

    def test_pipeline_runs_end_to_end(self, tmp_path, mock_raw_df):
        """Verify the full pipeline completes without errors on mock data."""
        # Write mock data to a temp CSV
        raw_path = tmp_path / "data" / "raw" / "test.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        mock_raw_df.to_csv(raw_path, index=False)

        config = {
            "data": {
                "raw": str(raw_path),
                "processed": str(tmp_path / "data" / "processed" / "clean.csv"),
                "inference": str(tmp_path / "data" / "inference"),
            },
            "model": {
                "output_dir": str(tmp_path / "models"),
                "ols_path": str(tmp_path / "models" / "ols.pkl"),
                "lasso_path": str(tmp_path / "models" / "lasso.pkl"),
                "scaler_path": str(tmp_path / "models" / "scaler.pkl"),
            },
            "train": {"test_size": 0.2, "seed": 42, "lasso_cv_folds": 3},
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
            "reports": {"output_dir": str(tmp_path / "reports")},
            "logging": {
                "log_file": str(tmp_path / "logs" / "test.log"),
                "level": "DEBUG",
            },
        }

        run_training_pipeline(config)

        # Verify artifacts were created
        assert (tmp_path / "models" / "ols.pkl").exists()
        assert (tmp_path / "models" / "lasso.pkl").exists()
        assert (tmp_path / "models" / "scaler.pkl").exists()
        assert (tmp_path / "data" / "processed" / "clean.csv").exists()
        assert (tmp_path / "reports" / "actual_vs_predicted.png").exists()

    def test_pipeline_fails_on_missing_data(self, tmp_path):
        """Verify pipeline raises FileNotFoundError for missing data."""
        config = {
            "data": {
                "raw": str(tmp_path / "nonexistent.csv"),
                "processed": str(tmp_path / "clean.csv"),
            },
            "features": {
                "target": "Points",
                "target_components": ["Goals", "Primary_Assists", "Secondary_Assists"],
                "explanatory": ["Faceoff_Win_Pct"],
                "controls": ["Icetime_Minutes", "Pos"],
                "leakage_columns": [],
            },
            "train": {"test_size": 0.2, "seed": 42, "lasso_cv_folds": 3},
            "model": {"output_dir": str(tmp_path)},
            "reports": {"output_dir": str(tmp_path)},
            "logging": {
                "log_file": str(tmp_path / "logs" / "test.log"),
                "level": "DEBUG",
            },
        }

        with pytest.raises(FileNotFoundError):
            run_training_pipeline(config)
