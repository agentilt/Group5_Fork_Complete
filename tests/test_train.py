"""Tests for src/train.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features import get_feature_preprocessor
from src.train import train_model


@pytest.fixture
def training_setup():
    """Build training data and preprocessor for tests."""
    np.random.seed(42)
    n = 80
    X = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
        "cat": np.random.choice(["A", "B", "C"], n),
        "bin_col": np.random.uniform(0, 100, n),
    })
    y = pd.Series(2 * X["num1"] + X["num2"] + np.random.randn(n) * 0.3)
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["bin_col"],
        categorical_onehot_cols=["cat"],
        numeric_passthrough_cols=["num1", "num2"],
        n_bins=3,
    )
    return X, y, preprocessor


class TestTrainModel:
    def test_returns_pipeline(self, training_setup):
        X, y, pp = training_setup
        result = train_model(X, y, pp, "regression")
        assert isinstance(result, Pipeline)

    def test_pipeline_can_predict(self, training_setup):
        X, y, pp = training_setup
        pipeline = train_model(X, y, pp, "regression")
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_regression_r2_positive(self, training_setup):
        X, y, pp = training_setup
        pipeline = train_model(X, y, pp, "regression")
        assert pipeline.score(X, y) > 0

    def test_classification_mode(self):
        np.random.seed(42)
        n = 60
        X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        y = pd.Series(np.random.choice([0, 1], n))
        pp = get_feature_preprocessor(numeric_passthrough_cols=["f1", "f2"])

        pipeline = train_model(X, y, pp, "classification")
        assert isinstance(pipeline, Pipeline)
        assert set(pipeline.predict(X)).issubset({0, 1})

    def test_pipeline_has_two_steps(self, training_setup):
        X, y, pp = training_setup
        pipeline = train_model(X, y, pp, "regression")
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "preprocessor"
        assert pipeline.steps[1][0] == "model"
