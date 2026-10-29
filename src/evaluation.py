"""Shared model evaluation and export helpers."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class Evaluation:
    model_name: str
    predictions: np.ndarray
    probabilities: np.ndarray | None
    metrics: dict[str, float]
    confusion: np.ndarray
    report: dict
    training_time: float


def evaluate_model(model_name, model, x_test, y_test, labels, training_time=0.0):
    predictions = np.asarray(model.predict(x_test))
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_test))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, labels=labels, average="binary", pos_label=labels[1], zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, labels=labels, average="binary", pos_label=labels[1], zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, labels=labels, average="binary", pos_label=labels[1], zero_division=0)
        ),
    }
    return Evaluation(
        model_name,
        predictions,
        probabilities,
        metrics,
        confusion_matrix(y_test, predictions, labels=labels),
        classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        training_time,
    )
