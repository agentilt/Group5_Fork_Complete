"""
Module: Data Cleaning & Preprocessing
--------------------------------------
Role: Compute target variable, select features, encode categoricals,
      handle missing values, and remove duplicates.
Input: Raw pandas DataFrame and pipeline configuration.
Output: Cleaned, feature-selected pandas DataFrame ready for training.
"""

import logging

import pandas as pd

from src.utils import DataValidationError


logger = logging.getLogger("nhl_pipeline")


def compute_target(df: pd.DataFrame, target_name: str,
                   components: list) -> pd.DataFrame:
    """
    Compute the target variable as the sum of its component columns.

    Why: Points are derived from Goals + Primary_Assists +
    Secondary_Assists. Computing this explicitly avoids hidden
    assumptions in downstream modules.

    Args:
        df: DataFrame containing the component columns.
        target_name: Name for the computed target column.
        components: List of column names to sum.

    Returns:
        DataFrame with the new target column added.

    Raises:
        DataValidationError: If any component column is missing.
    """
    missing = [col for col in components if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Cannot compute target '{target_name}': "
            f"missing columns {missing}"
        )

    df = df.copy()
    df[target_name] = sum(df[col] for col in components)
    logger.info(
        "Computed target '%s' from %s — mean: %.2f",
        target_name, components, df[target_name].mean()
    )
    return df


def select_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Select the final set of features and target from the DataFrame.

    Why: Feature selection is driven by domain knowledge — only
    underlying skill metrics and environmental controls are kept,
    while rate stats, expected-value metrics, and identifiers are
    excluded to avoid multicollinearity and data leakage.

    Args:
        df: DataFrame with target already computed.
        config: Pipeline configuration dictionary.

    Returns:
        DataFrame with only selected features and target.

    Raises:
        DataValidationError: If required features are missing.
    """
    features_config = config["features"]
    explanatory = features_config["explanatory"]
    controls = features_config["controls"]
    target = features_config["target"]

    selected = explanatory + controls + [target]

    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Cannot select features — missing columns: {missing}"
        )

    df_selected = df[selected].copy()
    logger.info(
        "Selected %d features + target — shape: %s",
        len(selected) - 1, df_selected.shape
    )
    return df_selected


def encode_categoricals(df: pd.DataFrame, column: str = "Pos",
                        baseline: str = "C") -> pd.DataFrame:
    """
    One-hot encode a categorical column, dropping the baseline category.

    Why: Position (C/D/L/R) is the only categorical feature. Encoding
    it as dummies with Center as baseline matches the analytical
    framework where all positions are compared to Centers.

    Args:
        df: DataFrame containing the categorical column.
        column: Name of the column to encode.
        baseline: Category to use as baseline (dropped in encoding).

    Returns:
        DataFrame with dummy columns replacing the original.
    """
    if column not in df.columns:
        logger.info("Column '%s' not found — skipping encoding", column)
        return df

    df = df.copy()
    df[column] = pd.Categorical(df[column])

    # Ensure baseline category is first so drop_first removes it
    if baseline in df[column].cat.categories:
        ordered = [baseline] + [
            c for c in df[column].cat.categories if c != baseline
        ]
        df[column] = df[column].cat.set_categories(ordered)

    df = pd.get_dummies(df, columns=[column], drop_first=True, dtype=int)
    logger.info("Encoded '%s' — baseline: '%s'", column, baseline)
    return df


def clean_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Full cleaning and preprocessing pipeline.

    Runs all steps: compute target, select features, drop missing
    values and duplicates, and encode categoricals.

    Args:
        df: Raw DataFrame.
        config: Pipeline configuration dictionary.

    Returns:
        Cleaned, feature-selected, encoded DataFrame.

    Raises:
        DataValidationError: If target cannot be computed or
            required features are missing.
    """
    features_config = config["features"]
    target_name = features_config["target"]
    components = features_config["target_components"]

    logger.info("Starting data cleaning and preprocessing")

    # Step 1: Compute target
    df = compute_target(df, target_name, components)

    # Step 2: Select features
    df = select_features(df, config)

    # Step 3: Drop missing values and duplicates
    initial_rows = len(df)
    df = df.dropna()
    after_na = len(df)
    df = df.drop_duplicates()
    final_rows = len(df)

    dropped_na = initial_rows - after_na
    dropped_dup = after_na - final_rows

    if dropped_na > 0:
        logger.warning("Dropped %d rows with missing values", dropped_na)
    if dropped_dup > 0:
        logger.warning("Dropped %d duplicate rows", dropped_dup)

    # Step 4: Encode categoricals
    df = encode_categoricals(df, column="Pos", baseline="C")

    logger.info("Cleaning complete — %d rows, %d columns", len(df), len(df.columns))
    return df
