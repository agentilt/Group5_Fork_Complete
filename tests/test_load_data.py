"""Tests for src/load_data.py"""

from pathlib import Path

import pandas as pd
import pytest

from src.load_data import load_raw_data


class TestLoadRawData:
    """Tests for the load_raw_data function."""

    def test_returns_dataframe(self, tmp_csv):
        df = load_raw_data(str(tmp_csv))
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, tmp_csv):
        df = load_raw_data(str(tmp_csv))
        assert len(df) == 50

    def test_preserves_all_columns(self, tmp_csv):
        df = load_raw_data(str(tmp_csv))
        assert "Name" in df.columns
        assert "Goals" in df.columns
        assert "Pos" in df.columns

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data("data/raw/nonexistent.csv")

    def test_data_not_empty(self, tmp_csv):
        df = load_raw_data(str(tmp_csv))
        assert not df.empty
