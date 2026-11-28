
"""
SignSense - Confusion Matrix Visualization

Creates and saves a confusion matrix for model predictions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

from .metrics import CLASS_NAMES


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name="model",
    output_dir="results"
):
    """
    Generate and save a confusion matrix.

    Args:
        y_true: Actual integer labels.
        y_pred: Predicted integer labels.
        model_name: Model name (GRU, LSTM, etc.).
        output_dir: Directory where the image is saved.

    Returns:
        The confusion matrix as a NumPy array.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths.")

    if len(y_true) == 0:
        raise ValueError("Cannot plot an empty test set.")

    if np.any((y_true < 0) | (y_true >= len(CLASS_NAMES))):
        raise ValueError("Invalid actual class labels.")

    if np.any((y_pred < 0) | (y_pred >= len(CLASS_NAMES))):
        raise ValueError("Invalid predicted class labels.")

    # 1. Calculate confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES))
    )

    # 2. Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create visualization
    fig, ax = plt.subplots(figsize=(13, 11))

    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    # 4. Add class labels
    positions = np.arange(len(CLASS_NAMES))

    ax.set(
        xticks=positions,
        yticks=positions,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"Confusion Matrix - {model_name.upper()}"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # 5. Display numerical values in cells
    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.tight_layout()

    # 6. Save image
    output_path = output_dir / f"confusion_matrix_{model_name}.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Confusion matrix saved: {output_path}")

    return cm
