
import numpy as np
from evaluation.benchmark import measure_latency

def test_measure_latency():

    X = np.random.default_rng(42).random(
        (20, 30, 144)
    ).astype(np.float32)

    # Simulated prediction function (15 classes)
    def fake_predict_proba(data):
        probabilities = np.ones((len(data), 15)) / 15
        return probabilities

    results = measure_latency(
        predict_proba=fake_predict_proba,
        X_test=X,
        warmup=2,
        repeats=5
    )

    print("\n--- Benchmark Test Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")

    assert results["latency_per_sample_ms"] >= 0
    assert results["throughput_samples_per_second"] > 0
    assert results["number_of_samples"] == 20

    print("\nBenchmark test passed!")


if __name__ == "__main__":
    test_measure_latency()
