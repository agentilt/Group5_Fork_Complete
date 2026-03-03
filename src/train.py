"""
Module: Model Training
----------------------
Role: Wrap a preprocessor and a model inside a sklearn Pipeline,
      fit on the training split, and return the fitted pipeline.
Input: Training features, target, a ColumnTransformer, and problem type.
Output: Fitted sklearn Pipeline.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                preprocessor, problem_type: str):
    """
    Build and fit a sklearn Pipeline (preprocessor + model).

    Why: Wrapping preprocessing and modeling in a single Pipeline
    guarantees that the same transformations applied during training
    are automatically applied at prediction time, preventing
    train-serve skew.

    Args:
        X_train:       Training feature DataFrame.
        y_train:       Training target Series.
        preprocessor:  Unfitted ColumnTransformer from features.py.
        problem_type:  'regression' or 'classification'.

    Returns:
        Fitted sklearn Pipeline.
    """
    if problem_type == "classification":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = LinearRegression()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    print(f"[train] Pipeline fitted — R² on train: "
          f"{pipeline.score(X_train, y_train):.4f}")
    return pipeline
