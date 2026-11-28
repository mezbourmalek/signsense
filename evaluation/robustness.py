
"""
SignSense - Robustness Evaluation

Tests model performance under controlled changes to input sequences.
"""

import numpy as np

from .metrics import compute_metrics


LEFT_HAND = slice(0, 63)
RIGHT_HAND = slice(63, 126)


def add_noise(X, noise_std=0.01, random_state=42):
    """
    Add Gaussian noise to available landmarks.

    All-zero landmarks are kept unchanged.
    """
    X = np.asarray(X, dtype=np.float32)
    X_new = X.copy()

    rng = np.random.default_rng(random_state)

    # A landmark is considered available if at least one
    # of its three coordinates is non-zero.
    landmarks = X_new.reshape(*X_new.shape[:2], 48, 3)
    available = np.any(landmarks != 0, axis=-1)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=landmarks.shape
    )

    landmarks += noise.astype(np.float32) * available[..., None]

    return X_new


def drop_frames(X, drop_rate=0.2, random_state=42):
    """
    Randomly remove a proportion of frames from each sequence.

    Removed frames are replaced with zeros.
    """
    X_new = np.asarray(X, dtype=np.float32).copy()

    if not 0 <= drop_rate < 1:
        raise ValueError("drop_rate must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    n_samples, n_frames, _ = X_new.shape
    n_drop = int(round(n_frames * drop_rate))

    for i in range(n_samples):
        if n_drop > 0:
            indices = rng.choice(
                n_frames,
                size=n_drop,
                replace=False
            )
            X_new[i, indices, :] = 0.0

    return X_new


def drop_hand(X, hand="right"):
    """
    Remove one hand by replacing its features with zeros.

    hand: 'left' or 'right'
    """
    X_new = np.asarray(X, dtype=np.float32).copy()

    if hand == "left":
        X_new[:, :, LEFT_HAND] = 0.0
    elif hand == "right":
        X_new[:, :, RIGHT_HAND] = 0.0
    else:
        raise ValueError("hand must be 'left' or 'right'.")

    return X_new


def change_speed(X, factor=1.2):
    """
    Apply a simple temporal speed change using nearest-neighbor sampling.

    factor > 1: faster sequence
    factor < 1: slower sequence
    """
    X = np.asarray(X, dtype=np.float32)

    if factor <= 0:
        raise ValueError("factor must be positive.")

    n_samples, n_frames, n_features = X.shape

    center = (n_frames - 1) / 2

    output_positions = np.arange(n_frames)

    source_positions = (
        center + (output_positions - center) * factor
    )

    source_positions = np.clip(
        source_positions,
        0,
        n_frames - 1
    )

    source_indices = np.rint(source_positions).astype(int)

    return X[:, source_indices, :].copy()


def robustness_report(
    predict_proba,
    X_test,
    y_test,
    model_name="model"
):
    """
    Evaluate model performance on original and modified test data.

    predict_proba:
        Function that receives X and returns probabilities (N, 15).
    """

    scenarios = {
        "original": X_test,
        "noise": add_noise(X_test, noise_std=0.01),
        "drop_frames_20": drop_frames(X_test, drop_rate=0.20),
        "drop_left_hand": drop_hand(X_test, hand="left"),
        "drop_right_hand": drop_hand(X_test, hand="right"),
        "faster_1.2": change_speed(X_test, factor=1.2),
        "slower_0.8": change_speed(X_test, factor=0.8)
    }

    results = {}

    baseline_accuracy = None

    for scenario_name, X_modified in scenarios.items():

        probabilities = np.asarray(
            predict_proba(X_modified)
        )

        metrics = compute_metrics(y_test, probabilities)

        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]

        if scenario_name == "original":
            baseline_accuracy = accuracy

        results[scenario_name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

    # Calculate performance drops relative to original data
    for scenario_name, values in results.items():

        values["accuracy_drop"] = (
            baseline_accuracy - values["accuracy"]
        )

    print(f"\nRobustness report: {model_name}")

    for scenario_name, values in results.items():
        print(
            f"{scenario_name:20s} "
            f"Accuracy={values['accuracy']:.4f} "
            f"F1={values['macro_f1']:.4f} "
            f"Accuracy drop={values['accuracy_drop']:.4f}"
        )

    return results
