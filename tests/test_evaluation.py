import numpy as np
from sklearn.dummy import DummyClassifier

from src.evaluation import evaluate_model


def test_metrics_and_confusion_matrix():
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1], [2], [3]], [3, 3, 7, 7])
    result = evaluate_model("dummy", model, [[4], [5]], [3, 7], (3, 7))
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1"}
    assert result.confusion.shape == (2, 2)
