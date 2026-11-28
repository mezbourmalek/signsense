
"""
SignSense - Keras Model Evaluation

Evaluates an already-trained Keras model on the fixed test set.

Usage:
    python -m evaluation.evaluate_keras \
        --arrays path/to/arrays \
        --model path/to/gru.keras \
        --name gru
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from .data_loader import load_test_data
from .metrics import compute_metrics
from .confusion_matrix import plot_confusion_matrix
from .robustness import robustness_report
from .benchmark import measure_latency


def load_keras_predict(model_path, batch_size=32):
    """
    Load a trained Keras model and return a prediction function.
    """

    model = keras.models.load_model(model_path)

    def predict_proba(X):
        output = model.predict(
            X,
            batch_size=batch_size,
            verbose=0
        )

        output = np.asarray(output)

        if output.ndim != 2 or output.shape[1] != 15:
            raise ValueError(
                f"Expected model output shape (N, 15), got {output.shape}"
            )

        # Accept probabilities if they are valid and sum to 1.
        is_probability = (
            np.all(output >= 0) and
            np.all(output <= 1) and
            np.allclose(output.sum(axis=1), 1.0, atol=1e-5)
        )

        # Otherwise interpret the output as logits.
        if not is_probability:
            shifted = output - np.max(
                output, axis=1, keepdims=True
            )
            exp_values = np.exp(shifted)
            output = exp_values / exp_values.sum(
                axis=1, keepdims=True
            )

        return output

    return predict_proba


def full_evaluation(
    model_path,
    arrays_dir,
    model_name="model",
    output_dir="results",
    batch_size=32
):
    """
    Run the complete evaluation pipeline for one model.
    """

    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load fixed test data
    X_test, y_test, groups_test, hand_frames_test = (
        load_test_data(arrays_dir)
    )

    # 2. Load trained model
    predict_proba = load_keras_predict(
        model_path,
        batch_size=batch_size
    )

    # 3. Predict on original test data
    probabilities = predict_proba(X_test)
    y_pred = np.argmax(probabilities, axis=1)

    # Save predictions for later model comparison
    predictions_path = output_dir / f"{model_name}_predictions.npz"

    np.savez_compressed(
        predictions_path,
        y_true=y_test,
        y_pred=y_pred,
        groups=groups_test
    )

    # 4. Calculate classification metrics
    metrics = compute_metrics(y_test, probabilities)

    # 5. Generate confusion matrix
    cm = plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        model_name=model_name,
        output_dir=output_dir
    )

    # 6. Run robustness tests
    robustness = robustness_report(
        predict_proba=predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name
    )

    # 7. Measure inference speed
    latency = measure_latency(
        predict_proba=predict_proba,
        X_test=X_test,
        warmup=3,
        repeats=10
    )

    # 8. Participant-level accuracy
    participant_accuracy = {}

    for participant in np.unique(groups_test):
        mask = groups_test == participant

        participant_accuracy[str(int(participant))] = float(
            np.mean(y_pred[mask] == y_test[mask])
        )

    # 9. Analyze errors according to hand-frame count
    hand_frame_groups = {}

    conditions = {
        "under_10": hand_frames_test < 10,
        "10_to_19": (
            (hand_frames_test >= 10) &
            (hand_frames_test < 20)
        ),
        "20_or_more": hand_frames_test >= 20
    }

    for group_name, mask in conditions.items():
        if np.any(mask):
            hand_frame_groups[group_name] = {
                "samples": int(np.sum(mask)),
                "accuracy": float(
                    np.mean(y_pred[mask] == y_test[mask])
                )
            }

    # 10. Build final report
    report = {
        "model_name": model_name,
        "test_samples": int(len(X_test)),
        "test_participants": [
            int(x) for x in np.unique(groups_test)
        ],
        "metrics": metrics,
        "participant_accuracy": participant_accuracy,
        "hand_frame_analysis": hand_frame_groups,
        "robustness": robustness,
        "latency": latency,
        "confusion_matrix": cm.tolist()
    }

    # 11. Save JSON report
    report_path = output_dir / f"{model_name}_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    print(f"\nEvaluation completed: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Report saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SignSense Keras model."
    )

    parser.add_argument(
        "--arrays",
        required=True,
        help="Path to the prepared arrays folder."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained .keras model."
    )

    parser.add_argument(
        "--name",
        default="model",
        help="Model name, for example gru or lstm."
    )

    parser.add_argument(
        "--out",
        default="results",
        help="Output directory."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    full_evaluation(
        model_path=args.model,
        arrays_dir=args.arrays,
        model_name=args.name,
        output_dir=args.out,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
