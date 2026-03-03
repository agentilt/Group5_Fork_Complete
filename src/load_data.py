"""
Module: Data Loader
-------------------
Role: Ingest raw NHL player data from CSV source.
Input: Path to CSV file (from config.yaml).
Output: pandas DataFrame containing raw player statistics.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils import load_csv


logger = logging.getLogger("nhl_pipeline")


def load_raw_data(raw_data_path: str) -> pd.DataFrame:
    """
    Load the raw NHL player statistics CSV file.

    Why: Isolating data loading in its own module allows the
    pipeline to swap data sources (CSV, API, database) without
    modifying downstream processing logic.

    Args:
        raw_data_path: File path to the raw CSV data.

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
    """
    path = Path(raw_data_path)
    logger.info("Loading raw data from %s", path)

    try:
        df = load_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Ensure the dataset is placed in data/raw/."
        )

    logger.info(
        "Raw data loaded — %d rows, %d columns", len(df), len(df.columns)
    )
    return df
