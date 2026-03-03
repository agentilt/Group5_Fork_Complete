"""Tests for src/train.py"""

import numpy as np
import pandas as pd
import pytest

from src.train import (
    scale_features,
    split_data,
    train_lasso,
    train_models,
    train_ols,
)


@pytest.fixture
def training_data():
    """Create simple training data for model tests."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
    })
    y = pd.Series(
        3 * X["feat_1"] + 2 * X["feat_2"] + np.random.randn(n) * 0.5,
        name="target",
    )
    return X, y


class TestSplitData:
    """Tests for the split_data function."""

    def test_correct_split_ratio(self, mock_clean_df):
        X_train, X_test, y_train, y_test = split_data(
            mock_clean_df, "Points", 0.2, 42
        )
        total = len(X_train) + len(X_test)
        assert total == len(mock_clean_df)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_no_overlap(self, mock_clean_df):
        X_train, X_test, _, _ = split_data(
            mock_clean_df, "Points", 0.2, 42
        )
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0

    def test_target_not_in_features(self, mock_clean_df):
        X_train, X_test, _, _ = split_data(
            mock_clean_df, "Points", 0.2, 42
        )
        assert "Points" not in X_train.columns
        assert "Points" not in X_test.columns

    def test_reproducible_with_seed(self, mock_clean_df):
        result_1 = split_data(mock_clean_df, "Points", 0.2, 42)
        result_2 = split_data(mock_clean_df, "Points", 0.2, 42)
        pd.testing.assert_frame_equal(result_1[0], result_2[0])


class TestScaleFeatures:
    """Tests for the scale_features function."""

    def test_train_mean_near_zero(self, mock_clean_df):
        X_train, X_test, _, _ = split_data(
            mock_clean_df, "Points", 0.2, 42
        )
        X_train_s, _, _ = scale_features(X_train, X_test)
        # Mean of scaled train should be ~0
        assert abs(X_train_s.mean().mean()) < 0.01

    def test_train_std_near_one(self, mock_clean_df):
        X_train, X_test, _, _ = split_data(
            mock_clean_df, "Points", 0.2, 42
        )
        X_train_s, _, _ = scale_features(X_train, X_test)
        assert abs(X_train_s.std().mean() - 1.0) < 0.1

    def test_returns_scaler(self, mock_clean_df):
        X_train, X_test, _, _ = split_data(
            mock_clean_df, "Points", 0.2, 42
        )
        _, _, scaler = scale_features(X_train, X_test)
        assert hasattr(scaler, "transform")

    def test_preserves_column_names(self, mock_clean_df):
        X_train, X_test, _, _ = split_data(
            mock_clean_df, "Points", 0.2, 42
        )
        X_train_s, X_test_s, _ = scale_features(X_train, X_test)
        assert list(X_train_s.columns) == list(X_train.columns)


class TestTrainOls:
    """Tests for the train_ols function."""

    def test_returns_fitted_model(self, training_data):
        X, y = training_data
        model = train_ols(X, y)
        assert hasattr(model, "predict")
        assert hasattr(model, "coef_")

    def test_predictions_correct_shape(self, training_data):
        X, y = training_data
        model = train_ols(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_positive_r2(self, training_data):
        X, y = training_data
        model = train_ols(X, y)
        assert model.score(X, y) > 0


class TestTrainLasso:
    """Tests for the train_lasso function."""

    def test_returns_fitted_model(self, training_data):
        X, y = training_data
        model = train_lasso(X, y, cv_folds=3, seed=42)
        assert hasattr(model, "predict")
        assert hasattr(model, "alpha_")

    def test_alpha_is_positive(self, training_data):
        X, y = training_data
        model = train_lasso(X, y, cv_folds=3, seed=42)
        assert model.alpha_ > 0

    def test_predictions_correct_shape(self, training_data):
        X, y = training_data
        model = train_lasso(X, y, cv_folds=3, seed=42)
        preds = model.predict(X)
        assert len(preds) == len(y)


class TestTrainModels:
    """Tests for the full train_models pipeline."""

    def test_returns_all_keys(self, mock_clean_df, sample_config):
        result = train_models(mock_clean_df, sample_config)
        expected_keys = {
            "ols", "lasso", "scaler", "feature_names",
            "X_train", "X_test", "y_train", "y_test",
        }
        assert set(result.keys()) == expected_keys

    def test_models_can_predict(self, mock_clean_df, sample_config):
        result = train_models(mock_clean_df, sample_config)
        preds = result["ols"].predict(result["X_test"])
        assert len(preds) == len(result["y_test"])

    def test_feature_names_match_columns(self, mock_clean_df, sample_config):
        result = train_models(mock_clean_df, sample_config)
        assert result["feature_names"] == list(result["X_train"].columns)
