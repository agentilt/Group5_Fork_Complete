"""
Module: Feature Engineering
---------------------------
Role: Build a sklearn ColumnTransformer that bins, encodes, and
      passes through features according to the SETTINGS configuration.
Input: Lists of column names and binning parameters.
Output: A fitted-ready ColumnTransformer (feature recipe).
"""

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Build a ColumnTransformer that applies three transformations:

        1. Quantile binning  — discretises continuous features into
           equal-frequency bins (useful for non-linear relationships).
        2. One-hot encoding  — converts categorical columns into
           binary dummy variables, ignoring unseen categories.
        3. Passthrough       — keeps numeric features untouched.

    Any column NOT listed in one of the three groups is dropped
    (remainder='drop'), which acts as implicit feature selection.

    Args:
        quantile_bin_cols:      Columns to bin into quantiles.
        categorical_onehot_cols: Categorical columns to one-hot encode.
        numeric_passthrough_cols: Numeric columns to pass through as-is.
        n_bins:                  Number of quantile bins (default 3).

    Returns:
        Unfitted ColumnTransformer ready to be placed in a Pipeline.
    """
    transformers = []

    if quantile_bin_cols:
        transformers.append((
            "quantile_bin",
            KBinsDiscretizer(
                n_bins=n_bins, encode="onehot-dense", strategy="quantile",
            ),
            quantile_bin_cols,
        ))

    if categorical_onehot_cols:
        transformers.append((
            "categorical_onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_onehot_cols,
        ))

    if numeric_passthrough_cols:
        transformers.append((
            "numeric_passthrough",
            "passthrough",
            numeric_passthrough_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")
