"""Tests for src/features.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor


class TestGetFeaturePreprocessor:
    def test_returns_column_transformer(self):
        ct = get_feature_preprocessor(
            numeric_passthrough_cols=["a"],
        )
        assert isinstance(ct, ColumnTransformer)

    def test_remainder_is_drop(self):
        ct = get_feature_preprocessor(
            numeric_passthrough_cols=["a"],
        )
        assert ct.remainder == "drop"

    def test_quantile_bin_transformer(self):
        ct = get_feature_preprocessor(
            quantile_bin_cols=["x"],
            n_bins=4,
        )
        names = [name for name, _, _ in ct.transformers]
        assert "quantile_bin" in names

    def test_onehot_transformer(self):
        ct = get_feature_preprocessor(
            categorical_onehot_cols=["cat"],
        )
        names = [name for name, _, _ in ct.transformers]
        assert "categorical_onehot" in names

    def test_passthrough_transformer(self):
        ct = get_feature_preprocessor(
            numeric_passthrough_cols=["num"],
        )
        names = [name for name, _, _ in ct.transformers]
        assert "numeric_passthrough" in names

    def test_fit_transform_works(self):
        """End-to-end: the preprocessor can fit and transform data."""
        np.random.seed(42)
        df = pd.DataFrame({
            "num": np.random.randn(20),
            "cat": np.random.choice(["A", "B"], 20),
            "bin_col": np.random.uniform(0, 100, 20),
            "extra": np.random.randn(20),  # should be dropped
        })
        ct = get_feature_preprocessor(
            quantile_bin_cols=["bin_col"],
            categorical_onehot_cols=["cat"],
            numeric_passthrough_cols=["num"],
            n_bins=3,
        )
        result = ct.fit_transform(df)
        # 3 bins (onehot) + 2 categories + 1 passthrough = 6 columns
        assert result.shape == (20, 6)

    def test_none_args_returns_empty_transformer(self):
        ct = get_feature_preprocessor()
        assert len(ct.transformers) == 0

    def test_handles_optional_lists(self):
        """Only the lists provided should generate transformers."""
        ct = get_feature_preprocessor(
            categorical_onehot_cols=["cat"],
        )
        assert len(ct.transformers) == 1
