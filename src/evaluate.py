"""
Module: Evaluation
------------------
Role: Compute performance metrics and generate diagnostic plots.
Input: Trained models, test data, and configuration.
Output: Metrics dictionary and saved plots/reports in reports/.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from statsmodels.stats.stattools import durbin_watson


logger = logging.getLogger("nhl_pipeline")


def compute_metrics(model, X_train, X_test, y_train, y_test) -> dict:
    """
    Compute regression performance metrics for a single model.

    Args:
        model: Fitted model with a predict method.
        X_train: Scaled training features.
        X_test: Scaled test features.
        y_train: Training target values.
        y_test: Test target values.

    Returns:
        Dictionary with R², MAE, and RMSE for train and test sets.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_rmse": float(
            np.sqrt(mean_squared_error(y_train, y_train_pred))
        ),
        "test_rmse": float(
            np.sqrt(mean_squared_error(y_test, y_test_pred))
        ),
    }


def compute_ols_summary(X_train, y_train, feature_names: list) -> str:
    """
    Generate a statsmodels OLS summary with robust standard errors.

    Why: P-values and significance levels from robust OLS help
    identify which features have statistically reliable effects,
    complementing the predictive focus of sklearn models.

    Args:
        X_train: Scaled training features.
        y_train: Training target values.
        feature_names: List of feature names.

    Returns:
        OLS summary string with HC3 robust standard errors.
    """
    X_const = sm.add_constant(np.array(X_train))
    ols_model = sm.OLS(np.array(y_train), X_const).fit(cov_type="HC3")
    return ols_model.summary().as_text()


def generate_diagnostic_plots(models: dict, X_test, y_test,
                              feature_names: list,
                              save_dir: str) -> None:
    """
    Generate and save diagnostic plots for model evaluation.

    Why: Visual diagnostics (residual plots, actual vs. predicted)
    reveal patterns that summary metrics alone cannot capture,
    such as heteroskedasticity or systematic prediction bias.

    Args:
        models: Dictionary of fitted models.
        X_test: Scaled test features.
        y_test: Test target values.
        feature_names: List of feature names.
        save_dir: Directory to save plot images.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    y_test_arr = np.array(y_test)
    colors = {"ols": "#0038A8", "lasso": "#C8102E"}

    # Actual vs Predicted for each model
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        color = colors.get(name, "#333333")

        ax.scatter(
            y_test_arr, y_pred, alpha=0.5, color=color, edgecolors="white"
        )
        ax.plot(
            [y_test_arr.min(), y_test_arr.max()],
            [y_test_arr.min(), y_test_arr.max()],
            "r--", lw=2, label="Perfect Prediction",
        )
        ax.set_title(f"{name.upper()}: Actual vs Predicted (R²={r2:.3f})")
        ax.set_xlabel("Actual Points")
        ax.set_ylabel("Predicted Points")
        ax.legend()

    plt.tight_layout()
    fig.savefig(save_path / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    logger.info("Saved actual vs predicted plot")

    # Residual diagnostics (using first model)
    first_model = list(models.values())[0]
    y_pred = first_model.predict(X_test)
    residuals = y_test_arr - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, color="#C8102E")
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_title("Residuals vs Fitted Values")
    axes[0].set_xlabel("Predicted Points")
    axes[0].set_ylabel("Residuals")

    sns.histplot(residuals, kde=True, ax=axes[1], color="purple")
    axes[1].set_title("Distribution of Residuals")

    plt.tight_layout()
    fig.savefig(save_path / "residual_diagnostics.png", dpi=150)
    plt.close(fig)
    logger.info("Saved residual diagnostic plots")

    # Durbin-Watson statistic
    dw_score = durbin_watson(residuals)
    logger.info("Durbin-Watson statistic: %.4f (ideal ≈ 2.0)", dw_score)


def generate_report(models: dict, X_train, X_test, y_train, y_test,
                    feature_names: list, config: dict) -> dict:
    """
    Generate a full evaluation report for all models.

    Args:
        models: Dictionary of fitted models.
        X_train: Scaled training features.
        X_test: Scaled test features.
        y_train: Training target.
        y_test: Test target.
        feature_names: List of feature names.
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary containing metrics for each model.
    """
    report_dir = config["reports"]["output_dir"]
    logger.info("Generating evaluation report")

    report = {}
    for name, model in models.items():
        metrics = compute_metrics(model, X_train, X_test, y_train, y_test)
        report[name] = metrics
        logger.info(
            "%s — Train R²: %.4f, Test R²: %.4f, Test MAE: %.2f",
            name.upper(), metrics["train_r2"], metrics["test_r2"],
            metrics["test_mae"],
        )

    # OLS statistical summary with robust standard errors
    try:
        ols_summary = compute_ols_summary(X_train, y_train, feature_names)
        summary_path = Path(report_dir) / "ols_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(ols_summary)
        logger.info("Saved OLS summary to %s", summary_path)
    except Exception as e:
        logger.warning("Could not generate OLS summary: %s", e)

    # Diagnostic plots
    generate_diagnostic_plots(
        models, X_test, y_test, feature_names, report_dir
    )

    return report
