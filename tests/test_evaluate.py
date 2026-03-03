"""Tests for src/evaluate.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.evaluate import (
    compute_metrics,
    compute_ols_summary,
    generate_report,
)


@pytest.fixture
def fitted_model():
    """Return a simple fitted linear model with train/test data."""
    np.random.seed(42)
    n = 80
    X_train = pd.DataFrame({
        "a": np.random.randn(n), "b": np.random.randn(n),
    })
    y_train = pd.Series(2 * X_train["a"] + X_train["b"] + np.random.randn(n) * 0.3)

    X_test = pd.DataFrame({
        "a": np.random.randn(20), "b": np.random.randn(20),
    })
    y_test = pd.Series(2 * X_test["a"] + X_test["b"] + np.random.randn(20) * 0.3)

    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_returns_all_metric_keys(self, fitted_model):
        model, X_train, X_test, y_train, y_test = fitted_model
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test)
        expected = {
            "train_r2", "test_r2", "train_mae", "test_mae",
            "train_rmse", "test_rmse",
        }
        assert set(metrics.keys()) == expected

    def test_r2_between_zero_and_one(self, fitted_model):
        model, X_train, X_test, y_train, y_test = fitted_model
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test)
        assert 0 < metrics["train_r2"] <= 1.0
        assert 0 < metrics["test_r2"] <= 1.0

    def test_mae_is_non_negative(self, fitted_model):
        model, X_train, X_test, y_train, y_test = fitted_model
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test)
        assert metrics["train_mae"] >= 0
        assert metrics["test_mae"] >= 0

    def test_rmse_is_non_negative(self, fitted_model):
        model, X_train, X_test, y_train, y_test = fitted_model
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test)
        assert metrics["train_rmse"] >= 0
        assert metrics["test_rmse"] >= 0


class TestComputeOlsSummary:
    """Tests for the compute_ols_summary function."""

    def test_returns_string(self, fitted_model):
        _, X_train, _, y_train, _ = fitted_model
        summary = compute_ols_summary(X_train, y_train, ["a", "b"])
        assert isinstance(summary, str)

    def test_contains_r_squared(self, fitted_model):
        _, X_train, _, y_train, _ = fitted_model
        summary = compute_ols_summary(X_train, y_train, ["a", "b"])
        assert "R-squared" in summary


class TestGenerateReport:
    """Tests for the generate_report function."""

    def test_returns_dict_with_model_names(self, fitted_model, tmp_path):
        model, X_train, X_test, y_train, y_test = fitted_model
        models = {"ols": model}
        config = {"reports": {"output_dir": str(tmp_path)}}

        report = generate_report(
            models, X_train, X_test, y_train, y_test, ["a", "b"], config
        )
        assert "ols" in report

    def test_saves_plots(self, fitted_model, tmp_path):
        model, X_train, X_test, y_train, y_test = fitted_model
        models = {"ols": model}
        config = {"reports": {"output_dir": str(tmp_path)}}

        generate_report(
            models, X_train, X_test, y_train, y_test, ["a", "b"], config
        )
        assert (tmp_path / "actual_vs_predicted.png").exists()
        assert (tmp_path / "residual_diagnostics.png").exists()

    def test_saves_ols_summary(self, fitted_model, tmp_path):
        model, X_train, X_test, y_train, y_test = fitted_model
        models = {"ols": model}
        config = {"reports": {"output_dir": str(tmp_path)}}

        generate_report(
            models, X_train, X_test, y_train, y_test, ["a", "b"], config
        )
        assert (tmp_path / "ols_summary.txt").exists()
