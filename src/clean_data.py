"""
Module: Data Cleaning
---------------------
Role: Compute target variable, remove leakage / identifier columns,
      and handle missing values and duplicates.
Input: Raw pandas DataFrame and the target column name.
Output: Cleaned pandas DataFrame.
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame,
                    target_column: str) -> pd.DataFrame:
    """
    Clean the raw NHL dataframe.

    Steps:
        1. Compute the target (Points = Goals + Primary_Assists +
           Secondary_Assists) if component columns are present.
        2. Drop identifier and target-leakage columns.
        3. Remove rows with missing values and duplicate rows.

    Args:
        df_raw: Raw DataFrame straight from load_data.
        target_column: Name of the target column to create/use.

    Returns:
        Cleaned DataFrame ready for feature engineering and modeling.
    """
    print("[clean_data] Starting data cleaning")
    df = df_raw.copy()

    # 1. Compute target from components if they exist
    components = ["Goals", "Primary_Assists", "Secondary_Assists"]
    if all(c in df.columns for c in components):
        df[target_column] = (
            df["Goals"] + df["Primary_Assists"] + df["Secondary_Assists"]
        )
        print(f"[clean_data] Computed '{target_column}' from {components}")

    # 2. Drop identifier and leakage columns
    leakage_cols = [
        "Rank", "Name", "Team",
        "Goals", "Assists", "Primary_Assists", "Secondary_Assists",
    ]
    to_drop = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"[clean_data] Dropped {len(to_drop)} leakage/identifier columns")

    # 3. Handle missing values and duplicates
    before = len(df)
    df = df.dropna().drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[clean_data] Dropped {dropped} rows (NaN or duplicates)")

    print(f"[clean_data] Cleaning complete — {len(df)} rows, {len(df.columns)} columns")
    return df
