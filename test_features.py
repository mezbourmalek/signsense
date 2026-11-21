"""Checks for features.py (plain asserts). Run: python test_features.py"""
import numpy as np
import pandas as pd

import config
import features

LEFT, RIGHT = config.LEFT_HAND_SLICE, config.RIGHT_HAND_SLICE
SHOULDER_COLS = [config.POSE_SLICE.start + 3 * config.POSE_INDICES.index(i) for i in (11, 12)]


def make_sample(n_frames):
    """Random sample whose shoulders are at x=0.4 and x=0.6 (width 0.2), y=0.5."""
    sample = np.random.default_rng(0).uniform(0.2, 0.8, (n_frames, config.FEATURES_PER_FRAME))
    sample = sample.astype(np.float32)
    sample[:, SHOULDER_COLS[0]:SHOULDER_COLS[0] + 3] = [0.4, 0.5, 0.0]
    sample[:, SHOULDER_COLS[1]:SHOULDER_COLS[1] + 3] = [0.6, 0.5, 0.0]
    return sample


def test_output_shape():
    for n in (1, 5, 30, 100):
        out = features.preprocess_sample(make_sample(n))
        assert out.shape == (config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME)
        assert out.dtype == np.float32 and not np.isnan(out).any()


def test_normalize():
    norm = features.normalize(make_sample(10))
    left = norm[:, SHOULDER_COLS[0]:SHOULDER_COLS[0] + 3]
    right = norm[:, SHOULDER_COLS[1]:SHOULDER_COLS[1] + 3]
    assert np.allclose(left, [-0.5, 0, 0]) and np.allclose(right, [0.5, 0, 0])


def test_normalize_invariance():
    """Moving the person or standing closer to the camera must not change the result."""
    base = make_sample(10)
    moved = base * 2.5 + 0.3
    assert np.allclose(features.normalize(base), features.normalize(moved), atol=1e-4)


def test_fill_short_gaps():
    s = make_sample(14)
    s[:, LEFT] = np.nan
    s[:, RIGHT] = np.nan
    s[0, LEFT], s[4, LEFT], s[13, LEFT] = 0.0, 4.0, 9.0
    out = features.fill_short_gaps(s, max_gap=5)
    assert np.allclose(out[1:4, LEFT], [[1.0], [2.0], [3.0]])   # gap of 3: filled
    assert np.isnan(out[5:13, LEFT]).all()                      # gap of 8: left alone
    assert np.isnan(out[:, RIGHT]).all()                        # other hand untouched


def test_missing_hand_becomes_zero():
    s = make_sample(20)
    s[:, RIGHT] = np.nan
    out = features.preprocess_sample(s)
    assert (out[:, RIGHT] == 0).all() and not (out[:, LEFT] == 0).all()


def test_resample():
    for n in (10, 100):
        s = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, config.FEATURES_PER_FRAME))
        out = features.resample(s)
        assert len(out) == 30 and out[0, 0] == 0 and out[-1, 0] == n - 1
        assert (np.diff(out[:, 0]) >= 0).all()


def test_real_samples():
    """Run the pipeline on the 50-sample extraction (skipped if it does not exist)."""
    folder = config.PROCESSED_DIR / "extracted_test"
    if not (folder / "frames.npy").exists():
        print("skipped: no extracted_test folder")
        return
    frames = np.load(folder / "frames.npy")
    index = pd.read_csv(folder / "index.csv")
    out = np.stack([features.preprocess_sample(frames[r.start:r.start + r.n_frames])
                    for r in index.itertuples()])
    assert out.shape == (len(index), 30, 144) and not np.isnan(out).any()
    print(f"real samples: {out.shape}, min {out.min():.2f}, max {out.max():.2f}, "
          f"zero values {(out == 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("All tests passed.")