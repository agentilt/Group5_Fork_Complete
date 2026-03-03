"""Tests for src/load_data.py"""

import pandas as pd
import pytest

from src.load_data import load_raw_data


class TestLoadRawData:
    def test_returns_dataframe(self, tmp_csv):
        df = load_raw_data(tmp_csv)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, tmp_csv):
        df = load_raw_data(tmp_csv)
        assert len(df) == 50

    def test_preserves_columns(self, tmp_csv):
        df = load_raw_data(tmp_csv)
        assert "Name" in df.columns
        assert "Goals" in df.columns

    def test_generates_dummy_when_missing(self, tmp_path):
        """If the CSV doesn't exist, a dummy is created automatically."""
        fake_path = tmp_path / "data" / "raw" / "missing.csv"
        df = load_raw_data(fake_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert fake_path.exists()  # dummy was saved

    def test_dummy_has_required_columns(self, tmp_path):
        fake_path = tmp_path / "dummy.csv"
        df = load_raw_data(fake_path)
        for col in ["Pos", "Icetime_Minutes", "Shot_Attempts",
                     "Goals", "Primary_Assists", "Secondary_Assists"]:
            assert col in df.columns
