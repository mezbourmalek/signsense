
"""
SignSense - Inference Benchmark

Measures model prediction time on test data.
"""

import time

import numpy as np


def measure_latency(
    predict_proba,
    X_test,
    warmup=3,
    repeats=10
):
    """
    Measure inference time.

    Args:
        predict_proba: Model prediction function.
        X_test: Test input data.
        warmup: Number of initial calls excluded from timing.
        repeats: Number of timed calls.

    Returns:
        Dictionary containing timing measurements.
    """

    if len(X_test) == 0:
        raise ValueError("X_test cannot be empty.")

    if warmup < 0 or repeats < 1:
        raise ValueError("Invalid warmup or repeats value.")

    # 1. Warm-up calls
    # These calls are excluded because the first prediction
    # may include initialization overhead.
    for _ in range(warmup):
        predict_proba(X_test)

    # 2. Measure prediction time
    timings = []

    for _ in range(repeats):
        start = time.perf_counter()

        predict_proba(X_test)

        end = time.perf_counter()

        timings.append(end - start)

    # 3. Calculate statistics
    average_time = float(np.mean(timings))
    median_time = float(np.median(timings))

    total_samples = len(X_test)

    latency_per_sample_ms = (
        average_time / total_samples
    ) * 1000

    throughput = total_samples / average_time

    return {
        "average_batch_time_seconds": average_time,
        "median_batch_time_seconds": median_time,
        "latency_per_sample_ms": latency_per_sample_ms,
        "throughput_samples_per_second": throughput,
        "number_of_samples": total_samples,
        "repeats": repeats
    }
