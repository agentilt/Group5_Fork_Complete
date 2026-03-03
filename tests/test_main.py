"""Tests for src/main.py — integration tests."""

from pathlib import Path

import pandas as pd
import pytest

from src.main import main, SETTINGS


class TestMainIntegration:
    """End-to-end integration tests using mock data."""

    def _make_config(self, tmp_path, mock_raw_df):
        """Write mock CSV and return a patched SETTINGS dict."""
        raw_path = tmp_path / "data" / "raw" / "test.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        mock_raw_df.to_csv(raw_path, index=False)

        import src.main as main_module
        original = dict(main_module.SETTINGS)

        main_module.SETTINGS.update({
            "raw_data_path": str(raw_path),
            "processed_data_path": str(tmp_path / "data" / "processed" / "clean.csv"),
            "model_path": str(tmp_path / "models" / "model.joblib"),
            "predictions_path": str(tmp_path / "data" / "inference" / "predictions.csv"),
        })
        return original

    def test_pipeline_end_to_end(self, tmp_path, mock_raw_df):
        import src.main as main_module
        original = self._make_config(tmp_path, mock_raw_df)

        try:
            main()
        finally:
            main_module.SETTINGS.update(original)

        assert (tmp_path / "models" / "model.joblib").exists()
        assert (tmp_path / "data" / "processed" / "clean.csv").exists()
        assert (tmp_path / "data" / "inference" / "predictions.csv").exists()

    def test_predictions_csv_has_prediction_column(self, tmp_path, mock_raw_df):
        import src.main as main_module
        original = self._make_config(tmp_path, mock_raw_df)

        try:
            main()
        finally:
            main_module.SETTINGS.update(original)

        preds = pd.read_csv(tmp_path / "data" / "inference" / "predictions.csv")
        assert "prediction" in preds.columns
