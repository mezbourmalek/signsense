
import numpy as np
from evaluation.metrics import compute_metrics

def test_compute_metrics():
    # One correct prediction for each of the 15 classes
    y_true = np.arange(15)

    probabilities = np.eye(15, dtype=float)

    results = compute_metrics(y_true, probabilities)

    assert results["accuracy"] == 1.0
    assert results["macro_f1"] == 1.0
    assert results["top_3_accuracy"] == 1.0
    assert results["ece"] == 0.0

    print("All metrics tests passed!")


if __name__ == "__main__":
    test_compute_metrics()
