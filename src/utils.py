"""
Module: Utilities
-----------------
Role: Shared helper functions for file I/O and model persistence.
Input: File paths, DataFrames, model objects.
Output: Loaded DataFrames, saved files, loaded models.
"""

from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If CSV file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    df = pd.read_csv(filepath)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save a DataFrame to a CSV file without the index.

    Args:
        df: DataFrame to save.
        filepath: Destination file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Serialize a model or pipeline to disk using joblib.

    Args:
        model: Trained model, pipeline, or transformer.
        filepath: Destination file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Deserialize a model or pipeline from disk.

    Args:
        filepath: Path to the serialized model.

    Returns:
        Loaded model or pipeline.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)
