"""Tests for src/evaluate.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate_model


@pytest.fixture
def regression_setup():
    """Create a simple fitted pipeline with train/test data."""
    np.random.seed(42)
    X_train = pd.DataFrame({"a": np.random.randn(80), "b": np.random.randn(80)})
    y_train = pd.Series(2 * X_train["a"] + X_train["b"])
    X_test = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
    y_test = pd.Series(2 * X_test["a"] + X_test["b"])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]).fit(X_train, y_train)

    return pipeline, X_test, y_test


class TestEvaluateModel:
    def test_returns_float(self, regression_setup):
        pipeline, X_test, y_test = regression_setup
        score = evaluate_model(pipeline, X_test, y_test, "regression")
        assert isinstance(score, float)

    def test_r2_near_one_for_perfect_data(self, regression_setup):
        pipeline, X_test, y_test = regression_setup
        score = evaluate_model(pipeline, X_test, y_test, "regression")
        assert score > 0.95

    def test_classification_returns_float(self):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(50)})
        y = pd.Series((X["a"] > 0).astype(int))

        from sklearn.linear_model import LogisticRegression
        pipeline = Pipeline([
            ("model", LogisticRegression()),
        ]).fit(X, y)

        score = evaluate_model(pipeline, X, y, "classification")
        assert isinstance(score, float)
        assert 0 <= score <= 1.0

    def test_score_between_bounds(self, regression_setup):
        pipeline, X_test, y_test = regression_setup
        score = evaluate_model(pipeline, X_test, y_test, "regression")
        assert -1.0 <= score <= 1.0  # R² can be negative for bad models
