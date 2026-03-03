"""
Module: Inference
-----------------
Role: Apply a trained pipeline to new / unseen data.
Input: Fitted model/pipeline and a DataFrame of new observations.
Output: DataFrame with a single column 'prediction', preserving index.
"""

import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions on new data using a fitted pipeline.

    Why: Keeping inference in its own module makes it easy to
    swap data sources or call this function from a serving layer
    without touching training logic.

    Args:
        model:   Fitted pipeline with a .predict() method.
        X_infer: DataFrame of new observations (same columns as
                 the training features).

    Returns:
        DataFrame with exactly one column named 'prediction'
        and the same index as X_infer.
    """
    predictions = model.predict(X_infer)
    return pd.DataFrame(
        {"prediction": predictions},
        index=X_infer.index,
    )
