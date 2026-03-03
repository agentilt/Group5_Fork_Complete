"""
Module: Main Pipeline
---------------------
Role: Orchestrate the full ML pipeline for traceability and
      robust error handling.
Usage: python src/main.py
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import save_csv, save_model
from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — SETTINGS DICTIONARY                           ║
# ║                                                                ║
# ║  ⚠️  UPDATE THIS BLOCK WHEN SWITCHING TO YOUR REAL DATASET ⚠️  ║
# ║  Every path, column name, and feature list must match your     ║
# ║  actual data. The current values are pre-configured for the    ║
# ║  NHL player stats dataset (or the dummy CSV fallback).         ║
# ╚══════════════════════════════════════════════════════════════════╝

SETTINGS = {
    # Set to True when using auto-generated dummy data.
    # Set to False once you plug in your real dataset.
    "is_example_config": False,

    "target_column": "Points",
    "problem_type": "regression",      # "regression" or "classification"
    "test_size": 0.2,
    "random_state": 42,

    # --- Paths ---
    "raw_data_path": "data/raw/nhl_player_stats.csv",
    "processed_data_path": "data/processed/clean.csv",
    "model_path": "models/model.joblib",
    "predictions_path": "data/inference/predictions.csv",

    # --- Feature recipe ---
    # Columns listed here must exist in the cleaned DataFrame.
    # The ColumnTransformer drops everything NOT listed (remainder='drop').
    "features": {
        "quantile_bin": [
            "Icetime_Minutes",
            "Shot_Attempts",
        ],
        "categorical_onehot": [
            "Pos",
        ],
        "numeric_passthrough": [
            "Faceoff_Win_Pct",
            "Takeaways",
            "Giveaways",
            "Shooting_Pct_On_Unblocked",
            "PIM_Drawn",
            "Pct_Shift_Starts_Offensive_Zone",
            "On_Ice_Corsi_Pct",
        ],
        "n_bins": 3,
    },
}


def main():
    """Run the full NHL Points Prediction Pipeline."""

    # ──────────────────────────────────────────
    # 0. CREATE DIRECTORIES
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 0: Creating directories")
    print("=" * 60)
    for d in ["data/raw", "data/processed", "data/inference",
              "models", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # 0b. EXAMPLE CONFIG CHECK
    # ──────────────────────────────────────────
    if SETTINGS["is_example_config"]:
        print("\n⚠️  WARNING: Running with EXAMPLE configuration!")
        print("   The pipeline will generate dummy data.")
        print("   Update SETTINGS to point to your real dataset.\n")

    # ──────────────────────────────────────────
    # 1. LOAD RAW DATA
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)
    df_raw = load_raw_data(Path(SETTINGS["raw_data_path"]))

    # ──────────────────────────────────────────
    # 2. CLEAN DATA
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Cleaning data")
    print("=" * 60)
    df_clean = clean_dataframe(df_raw, SETTINGS["target_column"])

    # ──────────────────────────────────────────
    # 3. SAVE PROCESSED CSV
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Saving processed data")
    print("=" * 60)
    save_csv(df_clean, Path(SETTINGS["processed_data_path"]))
    print(f"[main] Saved processed data to {SETTINGS['processed_data_path']}")

    # ──────────────────────────────────────────
    # 4. VALIDATE
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Validating data")
    print("=" * 60)
    all_feature_cols = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
    )
    required = [SETTINGS["target_column"]] + all_feature_cols
    validate_dataframe(df_clean, required)

    # ──────────────────────────────────────────
    # 5. TRAIN / TEST SPLIT (before any fitting)
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 5: Train / test split")
    print("=" * 60)
    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    try:
        stratify = y if SETTINGS["problem_type"] == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=stratify,
        )
    except ValueError:
        # Fallback if stratification fails (e.g., too few samples per class)
        print("[main] Stratification failed — falling back to random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
        )

    print(f"[main] Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # ──────────────────────────────────────────
    # 6. FAIL-FAST FEATURE CHECKS
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 6: Feature checks")
    print("=" * 60)
    missing = [c for c in all_feature_cols if c not in X_train.columns]
    if missing:
        raise ValueError(f"Configured feature columns missing from data: {missing}")

    for col in SETTINGS["features"]["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise TypeError(
                f"Column '{col}' listed in quantile_bin must be numeric, "
                f"got {X_train[col].dtype}"
            )
    print("[main] All feature checks passed")

    # ──────────────────────────────────────────
    # 7. BUILD FEATURE RECIPE (ColumnTransformer)
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 7: Building feature preprocessor")
    print("=" * 60)
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=SETTINGS["features"]["n_bins"],
    )
    print("[main] ColumnTransformer ready")

    # ──────────────────────────────────────────
    # 8. TRAIN MODEL (Pipeline: preprocess + model)
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 8: Training model")
    print("=" * 60)
    pipeline = train_model(
        X_train, y_train, preprocessor, SETTINGS["problem_type"]
    )

    # ──────────────────────────────────────────
    # 9. SAVE MODEL
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 9: Saving model")
    print("=" * 60)
    save_model(pipeline, Path(SETTINGS["model_path"]))
    print(f"[main] Saved model to {SETTINGS['model_path']}")

    # ──────────────────────────────────────────
    # 10. EVALUATE ON HELD-OUT TEST SET
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 10: Evaluating on test set")
    print("=" * 60)
    score = evaluate_model(pipeline, X_test, y_test, SETTINGS["problem_type"])
    print(f"[main] Final score: {score:.4f}")

    # ──────────────────────────────────────────
    # 11. INFERENCE ON EXAMPLE DATA
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 11: Running inference on example data")
    print("=" * 60)
    X_infer = X_test.head(5)
    preds_df = run_inference(pipeline, X_infer)
    print(preds_df)

    # ──────────────────────────────────────────
    # 12. SAVE PREDICTIONS
    # ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 12: Saving predictions")
    print("=" * 60)
    save_csv(preds_df, Path(SETTINGS["predictions_path"]))
    print(f"[main] Saved predictions to {SETTINGS['predictions_path']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
