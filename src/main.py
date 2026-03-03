"""
Module: Main Pipeline
---------------------
Role: Orchestrate the full ML pipeline with logging and error handling.
Usage: python src/main.py [--config CONFIG_PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    PipelineError,
    load_config,
    save_csv,
    save_model,
    setup_logging,
)
from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.train import train_models
from src.evaluate import generate_report
from src.infer import run_inference


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NHL Points Prediction Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--inference-only", action="store_true",
        help="Skip training and run inference only"
    )
    return parser.parse_args()


def run_training_pipeline(config: dict) -> None:
    """
    Execute the full training pipeline end-to-end.

    Why: A single orchestrator function ensures every step runs
    in the correct order with consistent logging, making the
    pipeline reproducible and traceable.

    Args:
        config: Pipeline configuration dictionary.
    """
    logger = logging.getLogger("nhl_pipeline")

    # Step 1: Load raw data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading raw data")
    logger.info("=" * 60)
    df_raw = load_raw_data(config["data"]["raw"])

    # Step 2: Clean, select features, and encode
    logger.info("=" * 60)
    logger.info("STEP 2: Cleaning and preprocessing data")
    logger.info("=" * 60)
    df_clean = clean_dataframe(df_raw, config)

    # Step 3: Validate
    logger.info("=" * 60)
    logger.info("STEP 3: Validating data")
    logger.info("=" * 60)
    validate_dataframe(df_clean, config)

    # Step 4: Save processed data
    logger.info("=" * 60)
    logger.info("STEP 4: Saving processed data")
    logger.info("=" * 60)
    save_csv(df_clean, Path(config["data"]["processed"]))

    # Step 5: Split, scale, and train models
    logger.info("=" * 60)
    logger.info("STEP 5: Training models")
    logger.info("=" * 60)
    results = train_models(df_clean, config)

    # Step 6: Evaluate models
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluating models")
    logger.info("=" * 60)
    models = {"ols": results["ols"], "lasso": results["lasso"]}
    generate_report(
        models,
        results["X_train"], results["X_test"],
        results["y_train"], results["y_test"],
        results["feature_names"], config,
    )

    # Step 7: Save artifacts
    logger.info("=" * 60)
    logger.info("STEP 7: Saving model artifacts")
    logger.info("=" * 60)
    save_model(results["scaler"], Path(config["model"]["scaler_path"]))
    for name in ("ols", "lasso"):
        save_model(results[name], Path(config["model"][f"{name}_path"]))

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


def main():
    """Entry point for the NHL Points Prediction Pipeline."""
    args = parse_args()

    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}))
    logger.info("NHL Points Prediction Pipeline started")

    try:
        if args.inference_only:
            logger.info("Running inference only")
            results = run_inference(config)
            if not results.empty:
                logger.info(
                    "Undervalued players:\n%s", results.to_string()
                )
        else:
            run_training_pipeline(config)

    except PipelineError as e:
        logger.error("Pipeline error: %s", e)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
