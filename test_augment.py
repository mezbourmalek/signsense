"""Checks for augment.py (plain asserts). Run: python test_augment.py"""
import numpy as np

import augment
import config
import features

LEFT, RIGHT = config.LEFT_HAND_SLICE, config.RIGHT_HAND_SLICE
SH = [config.POSE_SLICE.start + 3 * config.POSE_INDICES.index(i) for i in (11, 12)]


def make_sequence(seed=0):
    """Preprocessed-like sequence: left hand present (with a gap), right hand missing (zeros)."""
    raw = np.random.default_rng(seed).uniform(0.2, 0.8, (40, config.FEATURES_PER_FRAME)).astype(np.float32)
    raw[:, SH[0]:SH[0] + 3] = [0.4, 0.5, 0.0]
    raw[:, SH[1]:SH[1] + 3] = [0.6, 0.5, 0.0]
    raw[:, RIGHT] = np.nan
    raw[10:20, LEFT] = np.nan
    return features.preprocess_sample(raw)


def test_shape_dtype_no_nan():
    out = augment.augment_sequence(make_sequence(), np.random.default_rng(1))
    assert out.shape == (30, 144) and out.dtype == np.float32 and not np.isnan(out).any()


def test_missing_hand_stays_zero():
    seq = make_sequence()
    assert (seq[:, RIGHT] == 0).all()
    for fn in (lambda s: augment.scale(s, 1.1), lambda s: augment.shift(s, [0.1, -0.1, 0.05]),
               lambda s: augment.rotate(s, 10), lambda s: augment.change_speed(s, 1.2),
               lambda s: augment.add_noise(s, 0.01, np.random.default_rng(0)),
               lambda s: augment.augment_sequence(s, np.random.default_rng(3))):
        assert (fn(seq)[:, RIGHT] == 0).all()                  # never a phantom right hand


def test_zero_left_frames_stay_zero():
    seq = make_sequence()
    gone = (seq[:, LEFT] == 0).all(axis=1)
    assert gone.any()
    for out in (augment.scale(seq, 1.1), augment.shift(seq, [0.1, 0.1, 0.1]),
                augment.add_noise(seq, 0.05, np.random.default_rng(0))):
        assert (out[gone][:, LEFT] == 0).all()


def test_shoulders_unchanged_by_scale_shift_noise():
    seq = make_sequence()
    for out in (augment.scale(seq, 1.1), augment.shift(seq, [0.1, 0.1, 0.1]),
                augment.add_noise(seq, 0.05, np.random.default_rng(0))):
        for s in SH:
            assert np.array_equal(out[:, s:s + 3], seq[:, s:s + 3])


def test_rotation_keeps_shoulder_width_and_distances():
    seq = make_sequence()
    out = augment.rotate(seq, 10)
    width_before = np.linalg.norm(seq[:, SH[0]:SH[0] + 2] - seq[:, SH[1]:SH[1] + 2], axis=1)
    width_after = np.linalg.norm(out[:, SH[0]:SH[0] + 2] - out[:, SH[1]:SH[1] + 2], axis=1)
    assert np.allclose(width_before, width_after, atol=1e-4)
    assert np.allclose(np.linalg.norm(seq[:, 0:2], axis=1), np.linalg.norm(out[:, 0:2], axis=1), atol=1e-4)
    assert np.allclose(out[:, 2::3], seq[:, 2::3])             # z untouched


def test_speed_factor_one_is_identity_and_range():
    seq = make_sequence()
    assert np.array_equal(augment.change_speed(seq, 1.0), seq)
    for factor in (0.8, 1.25):
        out = augment.change_speed(seq, factor)
        assert out.shape == seq.shape
        assert all(any(np.array_equal(row, r) for r in seq) for row in out)   # only original frames


def test_input_not_modified_and_reproducible():
    seq = make_sequence()
    before = seq.copy()
    a = augment.augment_sequence(seq, np.random.default_rng(5))
    b = augment.augment_sequence(seq, np.random.default_rng(5))
    assert np.array_equal(seq, before) and np.array_equal(a, b)
    assert not np.array_equal(a, seq)


def test_augment_batch():
    X = np.stack([make_sequence(i) for i in range(4)])
    y = np.array([0, 1, 2, 3])
    X_aug, y_aug = augment.augment_batch(X, y, copies=3, seed=7)
    assert X_aug.shape == (12, 30, 144) and y_aug.tolist() == [0, 1, 2, 3] * 3
    X_aug2, _ = augment.augment_batch(X, y, copies=3, seed=7)
    assert np.array_equal(X_aug, X_aug2)
    assert (X_aug[:, :, RIGHT] == 0).all()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("All tests passed.")