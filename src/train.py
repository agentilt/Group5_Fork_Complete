"""
Module: Model Training
----------------------
Role: Split data, scale features, train models, and save artifacts.
Input: Cleaned pandas DataFrame (from clean_data) and configuration.
Output: Trained model objects (OLS and LassoCV) and fitted scaler.
"""

import logging

import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("nhl_pipeline")


def split_data(df: pd.DataFrame, target: str, test_size: float,
               seed: int) -> tuple:
    """
    Split DataFrame into train and test sets.

    Args:
        df: DataFrame with features and target.
        target: Name of the target column.
        test_size: Fraction of data reserved for testing.
        seed: Random state for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    logger.info(
        "Split data — train: %d rows, test: %d rows (%.0f%% test)",
        len(X_train), len(X_test), test_size * 100
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler fitted on training data only.

    Why: Scaling ensures features measured in different units
    (e.g., ice time in minutes vs. shooting percentage) are
    comparable, preventing large-magnitude features from dominating.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    feature_names = X_train.columns.tolist()
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    logger.info("Scaled %d features with StandardScaler", len(feature_names))
    return X_train_scaled, X_test_scaled, scaler


def train_ols(X_train, y_train):
    """
    Train an Ordinary Least Squares regression model.

    Why: OLS serves as the baseline model — it fits a linear
    relationship without regularization, providing interpretable
    coefficients for all features.

    Args:
        X_train: Scaled training features.
        y_train: Training target values.

    Returns:
        Fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    logger.info("OLS trained — R² on train: %.4f", train_r2)
    return model


def train_lasso(X_train, y_train, cv_folds: int = 5, seed: int = 42):
    """
    Train a Lasso regression with cross-validated alpha selection.

    Why: LassoCV adds L1 regularization that automatically shrinks
    unimportant feature coefficients toward zero, acting as an
    embedded feature selector and reducing overfitting risk.

    Args:
        X_train: Scaled training features.
        y_train: Training target values.
        cv_folds: Number of cross-validation folds.
        seed: Random state for reproducibility.

    Returns:
        Fitted LassoCV model.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    model = LassoCV(cv=kf, random_state=seed)
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    logger.info(
        "LassoCV trained — R² on train: %.4f, alpha: %.6f",
        train_r2, model.alpha_
    )
    return model


def train_models(df: pd.DataFrame, config: dict) -> dict:
    """
    Full training pipeline: split, scale, train all models.

    Args:
        df: Cleaned, encoded DataFrame from clean_data.
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with keys: ols, lasso, scaler, feature_names,
        X_train, X_test, y_train, y_test.
    """
    target = config["features"]["target"]
    test_size = config["train"]["test_size"]
    seed = config["train"]["seed"]
    cv_folds = config["train"]["lasso_cv_folds"]

    logger.info("Starting model training pipeline")

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size, seed
    )

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    feature_names = X_train_scaled.columns.tolist()

    # Train
    ols_model = train_ols(X_train_scaled, y_train)
    lasso_model = train_lasso(X_train_scaled, y_train, cv_folds, seed)

    logger.info("All models trained successfully")

    return {
        "ols": ols_model,
        "lasso": lasso_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
    }
