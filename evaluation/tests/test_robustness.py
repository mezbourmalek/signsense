
import numpy as np

from evaluation.robustness import (
    add_noise,
    drop_frames,
    drop_hand,
    change_speed
)

def test_robustness_functions():

    # Small synthetic dataset
    X = np.random.default_rng(42).random((10, 30, 144)).astype(np.float32)

    # Make some landmarks unavailable
    X[:, :, :3] = 0

    noisy = add_noise(X)
    dropped = drop_frames(X)
    left_removed = drop_hand(X, hand="left")
    right_removed = drop_hand(X, hand="right")
    faster = change_speed(X, factor=1.2)
    slower = change_speed(X, factor=0.8)

    # Check shapes
    for result in [
        noisy, dropped, left_removed,
        right_removed, faster, slower
    ]:
        assert result.shape == X.shape
        assert np.isfinite(result).all()

    # Check that original data was not modified
    assert X[:, :, :3].sum() == 0

    print("All robustness function tests passed!")


if __name__ == "__main__":
    test_robustness_functions()
