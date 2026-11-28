
"""
SignSense - Evaluation Metrics

Calculates classification metrics for trained Keras models.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


CLASS_NAMES = [
    "hello",
    "thankyou",
    "please",
    "yes",
    "no",
    "wait",
    "finish",
    "where",
    "go",
    "home",
    "drink",
    "book",
    "airplane",
    "car",
    "jump"
]


def top_k_accuracy(y_true, probabilities, k=3):
    """
    Checks whether the correct class is among the top k predictions.
    """

    top_k_predictions = np.argsort(
        probabilities, axis=1
    )[:, -k:]

    correct = [
        true_label in predicted_labels
        for true_label, predicted_labels
        in zip(y_true, top_k_predictions)
    ]

    return float(np.mean(correct))


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """
    Measures the difference between model confidence
    and actual accuracy.
    """

    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (confidences >= bin_edges[i]) &
                (confidences <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidences >= bin_edges[i]) &
                (confidences < bin_edges[i + 1])
            )

        if np.any(mask):
            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])

            ece += (
                np.mean(mask) *
                abs(bin_accuracy - bin_confidence)
            )

    return float(ece)


def compute_metrics(y_true, probabilities):
    """
    Calculate the main classification metrics.

    Args:
        y_true: Correct integer labels, shape (N,)
        probabilities: Model probabilities, shape (N, 15)

    Returns:
        Dictionary containing evaluation metrics.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must have shape (N, 15).")

    if probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("Expected probabilities for 15 classes.")

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities have different lengths.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to 1.")

    if np.any((y_true < 0) | (y_true >= len(CLASS_NAMES))):
        raise ValueError("y_true contains an invalid class label.")

    y_pred = np.argmax(probabilities, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top_3_accuracy": top_k_accuracy(
            y_true, probabilities, k=3
        ),
        "ece": expected_calibration_error(
            y_true, probabilities
        ),
        "classification_report": report
    }

    return results
