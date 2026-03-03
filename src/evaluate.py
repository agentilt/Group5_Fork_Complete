"""
Module: Evaluation
------------------
Role: Score a fitted pipeline on the held-out test set.
Input: Fitted model/pipeline, test features, test target, problem type.
Output: A single float metric (R² for regression, accuracy for classification).
"""

import pandas as pd
from sklearn.metrics import accuracy_score, r2_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   problem_type: str) -> float:
    """
    Evaluate a fitted pipeline on the test set and return one metric.

    Why: A single held-out score gives a fast, interpretable check
    of generalisation performance.  R² is used for regression
    (proportion of variance explained) and accuracy for classification.

    Args:
        model:        Fitted pipeline with a .predict() method.
        X_test:       Test feature DataFrame.
        y_test:       Test target Series.
        problem_type: 'regression' or 'classification'.

    Returns:
        Single float — R² score or accuracy.
    """
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        score = accuracy_score(y_test, y_pred)
        print(f"[evaluate] Test Accuracy: {score:.4f}")
    else:
        score = r2_score(y_test, y_pred)
        print(f"[evaluate] Test R²: {score:.4f}")

    return score
