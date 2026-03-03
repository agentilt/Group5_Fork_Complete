"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas DataFrame and validation criteria from config.
Output: True if valid, raises DataValidationError otherwise.
"""

import logging

import pandas as pd

from src.utils import DataValidationError


logger = logging.getLogger("nhl_pipeline")


def validate_not_empty(df: pd.DataFrame) -> None:
    """
    Raise if DataFrame is empty.

    Args:
        df: DataFrame to check.

    Raises:
        DataValidationError: If DataFrame has zero rows.
    """
    if df.empty:
        raise DataValidationError("DataFrame is empty.")


def validate_columns(df: pd.DataFrame, required_columns: list) -> None:
    """
    Raise if any required columns are missing from the DataFrame.

    Args:
        df: DataFrame to check.
        required_columns: List of column names that must be present.

    Raises:
        DataValidationError: If any required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}"
        )


def validate_no_nulls(df: pd.DataFrame) -> None:
    """
    Raise if DataFrame contains any null values.

    Args:
        df: DataFrame to check.

    Raises:
        DataValidationError: If any column contains null values.
    """
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        raise DataValidationError(
            f"Columns with null values: {dict(cols_with_nulls)}"
        )


def validate_numeric_columns(df: pd.DataFrame, columns: list) -> None:
    """
    Raise if specified columns are not numeric.

    Args:
        df: DataFrame to check.
        columns: List of column names expected to be numeric.

    Raises:
        DataValidationError: If any specified column is non-numeric.
    """
    non_numeric = [
        col for col in columns
        if col in df.columns
        and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise DataValidationError(
            f"Expected numeric but found non-numeric: {non_numeric}"
        )


def validate_no_leakage(df: pd.DataFrame, leakage_columns: list) -> None:
    """
    Raise if any target-leakage columns are still present.

    Why: Columns like Goals, Assists, and identifiers (Name, Team)
    must be removed before modeling to prevent data leakage and
    ensure the model learns from underlying metrics only.

    Args:
        df: DataFrame to check.
        leakage_columns: Columns that should have been dropped.

    Raises:
        DataValidationError: If any leakage column is found.
    """
    leaked = [col for col in leakage_columns if col in df.columns]
    if leaked:
        raise DataValidationError(
            f"Data leakage detected — remove: {leaked}"
        )


def validate_dataframe(df: pd.DataFrame, config: dict) -> bool:
    """
    Run all validation checks on the feature-selected DataFrame.

    Why: Strict validation gates prevent corrupted or incorrectly
    structured data from reaching the model, which could silently
    produce meaningless results.

    Args:
        df: DataFrame to validate.
        config: Pipeline configuration dictionary.

    Returns:
        True if all validations pass.

    Raises:
        DataValidationError: If any check fails.
    """
    features_config = config["features"]
    target = features_config["target"]
    explanatory = features_config["explanatory"]
    # Pos is categorical so exclude from numeric checks
    numeric_controls = [
        c for c in features_config["controls"] if c != "Pos"
    ]

    logger.info("Running validation checks")

    validate_not_empty(df)
    validate_columns(df, [target] + explanatory + numeric_controls)
    validate_no_nulls(df)
    validate_numeric_columns(df, explanatory + numeric_controls + [target])
    validate_no_leakage(df, features_config.get("leakage_columns", []))

    logger.info("All validation checks passed")
    return True
