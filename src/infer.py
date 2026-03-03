"""
Module: Inference
-----------------
Role: Apply trained models to new, unseen data for predictions.
Input: New player data (CSV), saved model and scaler artifacts.
Output: DataFrame with player names and predicted points.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils import load_csv, load_model
from src.clean_data import compute_target, encode_categoricals


logger = logging.getLogger("nhl_pipeline")


def prepare_inference_data(df: pd.DataFrame, config: dict,
                           scaler) -> tuple:
    """
    Prepare raw inference data through the same pipeline as training.

    Why: Inference data must undergo identical transformations
    (feature selection, encoding, scaling) as training data to
    ensure predictions are valid and comparable.

    Args:
        df: Raw DataFrame of new player data.
        config: Pipeline configuration dictionary.
        scaler: Fitted StandardScaler from training.

    Returns:
        Tuple of (scaled_features, player_names, actual_points).
        player_names and actual_points may be None if not available.
    """
    features_config = config["features"]
    components = features_config["target_components"]
    target = features_config["target"]
    explanatory = features_config["explanatory"]
    controls = features_config["controls"]

    # Preserve player identifiers before dropping columns
    names = df["Name"].copy() if "Name" in df.columns else None

    # Compute target if component columns are present
    if all(col in df.columns for col in components):
        df = compute_target(df, target, components)

    # Select features
    selected_cols = explanatory + controls
    if target in df.columns:
        selected_cols = selected_cols + [target]

    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Inference data missing columns: {missing}")

    df_selected = df[selected_cols].copy()

    # Encode categoricals (same as training)
    df_selected = encode_categoricals(
        df_selected, column="Pos", baseline="C"
    )

    # Separate target if present (for comparison)
    if target in df_selected.columns:
        y_actual = df_selected[target]
        X = df_selected.drop(columns=[target])
    else:
        y_actual = None
        X = df_selected

    # Scale using the saved training scaler
    feature_names = X.columns.tolist()
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_names,
        index=X.index,
    )

    return X_scaled, names, y_actual


def predict(model, X_scaled: pd.DataFrame) -> pd.Series:
    """
    Generate predictions from a fitted model.

    Args:
        model: Fitted model with a predict method.
        X_scaled: Scaled feature DataFrame.

    Returns:
        Series of predicted point values.
    """
    predictions = model.predict(X_scaled)
    return pd.Series(
        predictions, index=X_scaled.index, name="Predicted_Points"
    )


def identify_undervalued_players(names: pd.Series, y_actual: pd.Series,
                                 predictions: pd.Series,
                                 threshold: float = 5.0) -> pd.DataFrame:
    """
    Identify players whose predicted points exceed actual by a threshold.

    Why: The core business objective is finding undervalued talent —
    players whose underlying metrics suggest higher production than
    their actual point totals indicate.

    Args:
        names: Player names.
        y_actual: Actual point totals.
        predictions: Model-predicted point totals.
        threshold: Minimum difference to flag as undervalued.

    Returns:
        DataFrame of undervalued players sorted by point differential.
    """
    results = pd.DataFrame({
        "Name": names.values,
        "Actual_Points": y_actual.values,
        "Predicted_Points": predictions.values,
        "Difference": predictions.values - y_actual.values,
    })

    undervalued = results[results["Difference"] >= threshold].sort_values(
        "Difference", ascending=False
    )

    logger.info(
        "Found %d undervalued players (threshold: %.1f points)",
        len(undervalued), threshold
    )
    return undervalued


def run_inference(config: dict, model_name: str = "lasso") -> pd.DataFrame:
    """
    Run the full inference pipeline on new data.

    Args:
        config: Pipeline configuration dictionary.
        model_name: Which model to use ('ols' or 'lasso').

    Returns:
        DataFrame with predictions and undervalued player analysis.

    Raises:
        FileNotFoundError: If model artifacts or inference data missing.
    """
    logger.info("Starting inference pipeline")

    # Load saved artifacts
    model = load_model(Path(config["model"][f"{model_name}_path"]))
    scaler = load_model(Path(config["model"]["scaler_path"]))

    # Find inference CSVs
    inference_dir = Path(config["data"]["inference"])
    csv_files = list(inference_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {inference_dir} for inference."
        )

    all_results = []
    for csv_file in csv_files:
        logger.info("Running inference on %s", csv_file.name)
        df = load_csv(csv_file)

        X_scaled, names, y_actual = prepare_inference_data(
            df, config, scaler
        )
        predictions = predict(model, X_scaled)

        if y_actual is not None and names is not None:
            undervalued = identify_undervalued_players(
                names, y_actual, predictions
            )
            all_results.append(undervalued)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
