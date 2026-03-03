"""Tests for src/utils.py"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.utils import load_csv, load_model, save_csv, save_model


class TestLoadCsv:
    def test_loads_dataframe(self, tmp_csv):
        df = load_csv(tmp_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv(Path("nonexistent.csv"))

    def test_preserves_columns(self, tmp_csv):
        df = load_csv(tmp_csv)
        assert "Name" in df.columns


class TestSaveCsv:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        fp = tmp_path / "out.csv"
        save_csv(df, fp)
        assert fp.exists()

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        fp = tmp_path / "nested" / "dir" / "out.csv"
        save_csv(df, fp)
        assert fp.exists()

    def test_no_index_column(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        fp = tmp_path / "out.csv"
        save_csv(df, fp)
        loaded = pd.read_csv(fp)
        assert "Unnamed: 0" not in loaded.columns

    def test_roundtrip(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        fp = tmp_path / "out.csv"
        save_csv(df, fp)
        pd.testing.assert_frame_equal(df, pd.read_csv(fp))


class TestModelPersistence:
    def test_roundtrip(self, tmp_path):
        model = LinearRegression().fit([[1], [2], [3]], [1, 2, 3])
        fp = tmp_path / "model.joblib"
        save_model(model, fp)
        loaded = load_model(fp)
        assert loaded.predict([[4]])[0] == pytest.approx(4.0)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model(Path("nonexistent.joblib"))

    def test_creates_parent_dirs(self, tmp_path):
        model = LinearRegression().fit([[1]], [1])
        fp = tmp_path / "nested" / "model.joblib"
        save_model(model, fp)
        assert fp.exists()
