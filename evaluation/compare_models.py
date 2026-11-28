
"""
SignSense - Model Comparison

Compares two already-evaluated models on the same fixed test samples.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sklearn.metrics import accuracy_score, f1_score


N_CLASSES = 15


def load_predictions(path):
    """Load saved predictions and their test-set identifiers."""

    data = np.load(path)

    required = {"y_true", "y_pred", "groups"}

    if not required.issubset(data.files):
        raise ValueError(f"{path} is missing required arrays.")

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    groups = data["groups"]

    if y_true.ndim != 1 or y_pred.ndim != 1 or groups.ndim != 1:
        raise ValueError("y_true, y_pred and groups must be 1D arrays.")

    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("Prediction arrays have inconsistent lengths.")

    if not np.all(np.isin(y_true, range(N_CLASSES))):
        raise ValueError("y_true contains invalid class labels.")

    if not np.all(np.isin(y_pred, range(N_CLASSES))):
        raise ValueError("y_pred contains invalid class labels.")

    return {
        "y_true": data["y_true"],
        "y_pred": data["y_pred"],
        "groups": data["groups"]
    }


def bootstrap_difference(
    y_true,
    pred_a,
    pred_b,
    metric="accuracy",
    n_bootstrap=2000,
    random_state=42
):
    """
    Estimate the paired difference between two models.

    Difference is calculated as:
        Model A metric - Model B metric

    Returns the observed difference and a percentile 95% CI.
    """

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    if not (
        len(y_true) == len(pred_a) == len(pred_b)
    ):
        raise ValueError("Prediction arrays have different lengths.")

    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty test set.")

    if metric == "accuracy":
        metric_fn = accuracy_score
    elif metric == "macro_f1":
        metric_fn = lambda y, p: f1_score(
            y,
            p,
            labels=range(N_CLASSES),
            average="macro",
            zero_division=0
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    observed_difference = (
        metric_fn(y_true, pred_a) -
        metric_fn(y_true, pred_b)
    )

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    differences = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)

        score_a = metric_fn(
            y_true[indices],
            pred_a[indices]
        )

        score_b = metric_fn(
            y_true[indices],
            pred_b[indices]
        )

        differences.append(score_a - score_b)

    lower, upper = np.percentile(
        differences,
        [2.5, 97.5]
    )

    return {
        "observed_difference": float(observed_difference),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper)
    }


def compare_models(
    report_a_path,
    predictions_a_path,
    report_b_path,
    predictions_b_path,
    output_dir="results"
):
    """Compare two evaluated models and save comparison reports."""

    # 1. Load evaluation reports
    with open(report_a_path, encoding="utf-8") as f:
        report_a = json.load(f)

    with open(report_b_path, encoding="utf-8") as f:
        report_b = json.load(f)

    name_a = report_a["model_name"]
    name_b = report_b["model_name"]

    if name_a == name_b:
        raise ValueError("Model names must be different.")

    # 2. Load per-sample predictions
    pred_a = load_predictions(predictions_a_path)
    pred_b = load_predictions(predictions_b_path)

    # 3. Ensure both models used exactly the same test samples
    if not np.array_equal(
        pred_a["y_true"], pred_b["y_true"]
    ):
        raise ValueError("The models have different true labels or sample order.")

    if not np.array_equal(
        pred_a["groups"], pred_b["groups"]
    ):
        raise ValueError("The models were evaluated on different participants or sample order.")

    y_true = pred_a["y_true"]
    y_pred_a = pred_a["y_pred"]
    y_pred_b = pred_b["y_pred"]

    if len(y_true) != report_a["test_samples"]:
        raise ValueError("Prediction count does not match report A.")

    if len(y_true) != report_b["test_samples"]:
        raise ValueError("Prediction count does not match report B.")

    # 4. Extract main metrics
    metrics = ["accuracy", "macro_f1"]

    comparison = {
        "models": [name_a, name_b],
        "test_samples": int(len(y_true)),
        "metrics": {}
    }

    for metric in metrics:
        value_a = report_a["metrics"][metric]
        value_b = report_b["metrics"][metric]

        comparison["metrics"][metric] = {
            name_a: value_a,
            name_b: value_b,
            "difference_a_minus_b": float(value_a - value_b)
        }

    # 5. Paired bootstrap confidence intervals
    comparison["paired_bootstrap"] = {}

    for metric in metrics:
        comparison["paired_bootstrap"][metric] = (
            bootstrap_difference(
                y_true,
                y_pred_a,
                y_pred_b,
                metric=metric
            )
        )

    # 6. Compare inference speed
    latency_a = report_a["latency"]["latency_per_sample_ms"]
    latency_b = report_b["latency"]["latency_per_sample_ms"]

    comparison["latency_ms_per_sample"] = {
        name_a: latency_a,
        name_b: latency_b,
        "faster_model": (
            name_a if latency_a < latency_b else name_b
        )
    }

    # 7. Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "model_comparison.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            comparison,
            f,
            indent=4,
            ensure_ascii=False
        )

    csv_path = output_dir / "model_comparison.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            name_a,
            name_b,
            "Difference_A_minus_B"
        ])

        for metric in metrics:
            values = comparison["metrics"][metric]

            writer.writerow([
                metric,
                values[name_a],
                values[name_b],
                values["difference_a_minus_b"]
            ])

        writer.writerow([
            "Latency_ms_per_sample",
            latency_a,
            latency_b,
            latency_a - latency_b
        ])

    print("\nModel comparison completed.")

    for metric in metrics:
        values = comparison["metrics"][metric]

        print(
            f"{metric}: "
            f"{name_a}={values[name_a]:.4f}, "
            f"{name_b}={values[name_b]:.4f}"
        )

        ci = comparison["paired_bootstrap"][metric]

        print(
            f"  Difference ({name_a} - {name_b}): "
            f"{ci['observed_difference']:.4f}"
        )

        print(
            f"  95% CI: "
            f"[{ci['ci_95_lower']:.4f}, "
            f"{ci['ci_95_upper']:.4f}]"
        )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare two SignSense models."
    )

    parser.add_argument("--report-a", required=True)
    parser.add_argument("--predictions-a", required=True)

    parser.add_argument("--report-b", required=True)
    parser.add_argument("--predictions-b", required=True)

    parser.add_argument("--out", default="results")

    args = parser.parse_args()

    compare_models(
        report_a_path=args.report_a,
        predictions_a_path=args.predictions_a,
        report_b_path=args.report_b,
        predictions_b_path=args.predictions_b,
        output_dir=args.out
    )


if __name__ == "__main__":
    main()
