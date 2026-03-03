"""Tests for src/infer.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.infer import run_inference


@pytest.fixture
def simple_pipeline():
    """A simple fitted pipeline for inference tests."""
    np.random.seed(42)
    X = pd.DataFrame({"a": np.random.randn(30), "b": np.random.randn(30)})
    y = pd.Series(X["a"] + X["b"])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]).fit(X, y)


class TestRunInference:
    def test_returns_dataframe(self, simple_pipeline):
        X_infer = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = run_inference(simple_pipeline, X_infer)
        assert isinstance(result, pd.DataFrame)

    def test_single_prediction_column(self, simple_pipeline):
        X_infer = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = run_inference(simple_pipeline, X_infer)
        assert list(result.columns) == ["prediction"]

    def test_preserves_index(self, simple_pipeline):
        X_infer = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            index=[10, 20],
        )
        result = run_inference(simple_pipeline, X_infer)
        assert list(result.index) == [10, 20]

    def test_correct_row_count(self, simple_pipeline):
        X_infer = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = run_inference(simple_pipeline, X_infer)
        assert len(result) == 3

    def test_predictions_are_numeric(self, simple_pipeline):
        X_infer = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = run_inference(simple_pipeline, X_infer)
        assert pd.api.types.is_numeric_dtype(result["prediction"])
