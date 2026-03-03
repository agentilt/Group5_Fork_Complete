"""
Module: Data Validation
-----------------------
Role: Fail-fast quality gate — check that the DataFrame is usable
      before it reaches the modeling step.
Input: pandas DataFrame and list of required column names.
Output: True if valid, raises ValueError otherwise.
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame,
                       required_columns: list) -> bool:
    """
    Validate the DataFrame and fail fast for obvious issues.

    Checks performed:
        1. DataFrame is not empty.
        2. All required columns are present.
        3. No remaining null values in required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: If any check fails.
    """
    print("[validate] Running validation checks")

    # 1. Not empty
    if df.empty:
        raise ValueError("Validation failed: DataFrame is empty.")

    # 2. Required columns exist
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed: missing required columns {missing}"
        )

    # 3. No nulls in required columns
    null_counts = df[required_columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        raise ValueError(
            f"Validation failed: nulls found in {dict(cols_with_nulls)}"
        )

    print("[validate] All checks passed")
    return True
