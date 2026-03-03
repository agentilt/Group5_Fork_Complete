"""
Module: Utilities
-----------------
Role: Shared utility functions for logging setup, file I/O, and model persistence.
Input: Various (file paths, config dictionaries, DataFrames, models).
Output: Various (loggers, DataFrames, loaded models).
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml


class PipelineError(Exception):
    """Custom exception for pipeline-level failures."""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


def setup_logging(logging_config: dict) -> logging.Logger:
    """
    Configure and return a logger based on the provided config.

    Why: Centralizing logging setup ensures consistent formatting
    across all pipeline modules and enables persistent log files
    for debugging production runs.

    Args:
        logging_config: Dictionary with keys 'log_file', 'format',
                       'datefmt', and 'level'.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger("nhl_pipeline")

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    log_file = logging_config.get("log_file", "logs/pipeline.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = logging_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_level = getattr(
        logging, logging_config.get("level", "INFO").upper(), logging.INFO
    )

    formatter = logging.Formatter(log_format, datefmt=date_format)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load YAML configuration file.

    Why: Centralizing configuration avoids hard-coded paths and
    parameters scattered across modules, making the pipeline
    reproducible and easy to reconfigure.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        PipelineError: If config file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise PipelineError(f"Failed to parse config file: {e}") from e

    return config


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Why: Wrapping CSV loading in a utility ensures consistent
    error handling and logging across all data ingestion points.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If CSV file does not exist.
    """
    logger = logging.getLogger("nhl_pipeline")
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded CSV from %s — shape: %s", filepath, df.shape)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        filepath: Destination file path.
    """
    logger = logging.getLogger("nhl_pipeline")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Saved CSV to %s — shape: %s", filepath, df.shape)


def save_model(model, filepath: Path) -> None:
    """
    Serialize a model or scaler to disk using joblib.

    Args:
        model: Trained model or fitted transformer.
        filepath: Destination file path.
    """
    logger = logging.getLogger("nhl_pipeline")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Saved model to %s", filepath)


def load_model(filepath: Path):
    """
    Deserialize a model or scaler from disk.

    Args:
        filepath: Path to the serialized model.

    Returns:
        Loaded model or transformer.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    logger = logging.getLogger("nhl_pipeline")
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath)
    logger.info("Loaded model from %s", filepath)
    return model
