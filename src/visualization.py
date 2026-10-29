"""Matplotlib figures used by the Streamlit interface."""

import matplotlib.pyplot as plt
import numpy as np


def plot_samples(images, labels, count=8):
    count = min(count, len(images))
    fig, axes = plt.subplots(1, count, figsize=(1.8 * count, 2.2))
    axes = np.atleast_1d(axes)
    for axis, image, label in zip(axes, images[:count], labels[:count]):
        axis.imshow(image, cmap="gray")
        axis.set_title(str(label))
        axis.axis("off")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(evaluations):
    fig, axes = plt.subplots(1, len(evaluations), figsize=(5 * len(evaluations), 4))
    axes = np.atleast_1d(axes)
    for axis, evaluation in zip(axes, evaluations):
        image = axis.imshow(evaluation.confusion, cmap="Blues")
        axis.set_title(evaluation.model_name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for (row, col), value in np.ndenumerate(evaluation.confusion):
            axis.text(col, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_accuracy_comparison(evaluations):
    names = [evaluation.model_name for evaluation in evaluations]
    values = [evaluation.metrics["accuracy"] for evaluation in evaluations]
    fig, axis = plt.subplots(figsize=(6, 3.5))
    bars = axis.bar(names, values, color=["#53d8fb", "#8aa4c8"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Model accuracy comparison")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
    fig.tight_layout()
    return fig
