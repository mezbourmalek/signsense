
import json
import numpy as np
from pathlib import Path

from evaluation.compare_models import compare_models


def test_compare_models():

    test_dir = Path("evaluation/tests/temp_comparison")
    test_dir.mkdir(parents=True, exist_ok=True)

    output_dir = test_dir / "output"

    # Create synthetic test labels
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Fake predictions for Model A
    y_pred_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Fake predictions for Model B
    y_pred_b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0])

    groups = np.array([25571] * 4 + [26734] * 3 + [61333] * 3)

    # Create fake evaluation reports
    report_a = {
        "model_name": "GRU_test",
        "test_samples": 10,
        "metrics": {
            "accuracy": 1.0,
            "macro_f1": 1.0
        },
        "latency": {
            "latency_per_sample_ms": 2.0
        }
    }

    report_b = {
        "model_name": "LSTM_test",
        "test_samples": 10,
        "metrics": {
            "accuracy": 0.9,
            "macro_f1": 0.9
        },
        "latency": {
            "latency_per_sample_ms": 3.0
        }
    }

    # Save reports
    report_a_path = test_dir / "gru_report.json"
    report_b_path = test_dir / "lstm_report.json"

    with open(report_a_path, "w", encoding="utf-8") as f:
        json.dump(report_a, f)

    with open(report_b_path, "w", encoding="utf-8") as f:
        json.dump(report_b, f)

    # Save predictions
    pred_a_path = test_dir / "gru_predictions.npz"
    pred_b_path = test_dir / "lstm_predictions.npz"

    np.savez_compressed(
        pred_a_path,
        y_true=y_true,
        y_pred=y_pred_a,
        groups=groups
    )

    np.savez_compressed(
        pred_b_path,
        y_true=y_true,
        y_pred=y_pred_b,
        groups=groups
    )

    # Run comparison
    result = compare_models(
        report_a_path,
        pred_a_path,
        report_b_path,
        pred_b_path,
        output_dir=output_dir
    )

    # Verify results
    assert result["models"] == ["GRU_test", "LSTM_test"]
    assert result["test_samples"] == 10
    assert result["latency_ms_per_sample"]["faster_model"] == "GRU_test"

    assert (output_dir / "model_comparison.json").exists()
    assert (output_dir / "model_comparison.csv").exists()

    print("\nAll model comparison tests passed!")


if __name__ == "__main__":
    test_compare_models()
