"""Tests for src/utils.py"""

import logging
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.utils import (
    DataValidationError,
    PipelineError,
    load_config,
    load_csv,
    load_model,
    save_csv,
    save_model,
    setup_logging,
)


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_returns_logger(self, tmp_path):
        config = {"log_file": str(tmp_path / "test.log"), "level": "INFO"}
        logger = logging.getLogger("nhl_pipeline")
        logger.handlers.clear()

        result = setup_logging(config)
        assert isinstance(result, logging.Logger)

    def test_creates_log_directory(self, tmp_path):
        log_file = tmp_path / "subdir" / "test.log"
        config = {"log_file": str(log_file)}
        logger = logging.getLogger("nhl_pipeline")
        logger.handlers.clear()

        setup_logging(config)
        assert log_file.parent.exists()

    def test_no_duplicate_handlers(self, tmp_path):
        config = {"log_file": str(tmp_path / "test.log")}
        logger = logging.getLogger("nhl_pipeline")
        logger.handlers.clear()

        setup_logging(config)
        handler_count = len(logger.handlers)
        # Calling again should not add more handlers
        setup_logging(config)
        assert len(logger.handlers) == handler_count


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("data:\n  raw: 'test.csv'\n")

        config = load_config(str(config_file))
        assert config["data"]["raw"] == "test.csv"

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml")

        with pytest.raises(PipelineError):
            load_config(str(config_file))


class TestLoadCsv:
    """Tests for the load_csv function."""

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
        assert "Goals" in df.columns


class TestSaveCsv:
    """Tests for the save_csv function."""

    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        filepath = tmp_path / "output.csv"

        save_csv(df, filepath)
        assert filepath.exists()

    def test_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        filepath = tmp_path / "nested" / "dir" / "output.csv"

        save_csv(df, filepath)
        assert filepath.exists()

    def test_roundtrip_content(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        filepath = tmp_path / "output.csv"

        save_csv(df, filepath)
        loaded = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(df, loaded)


class TestModelPersistence:
    """Tests for save_model and load_model."""

    def test_save_and_load_roundtrip(self, tmp_path):
        model = LinearRegression()
        model.fit([[1], [2], [3]], [1, 2, 3])
        filepath = tmp_path / "model.pkl"

        save_model(model, filepath)
        loaded = load_model(filepath)

        assert loaded.predict([[4]])[0] == pytest.approx(4.0)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model(Path("nonexistent.pkl"))

    def test_creates_parent_dirs(self, tmp_path):
        model = LinearRegression()
        model.fit([[1]], [1])
        filepath = tmp_path / "nested" / "model.pkl"

        save_model(model, filepath)
        assert filepath.exists()


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_pipeline_error(self):
        with pytest.raises(PipelineError, match="test"):
            raise PipelineError("test")

    def test_data_validation_error(self):
        with pytest.raises(DataValidationError, match="invalid"):
            raise DataValidationError("invalid")
