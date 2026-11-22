"""Data augmentation for SignSense. Apply to TRAINING samples only.

Works on preprocessed sequences [30, 144] (the output of features.preprocess_sample,
i.e. what is stored in X.npy). Rules:
- A hand that is not visible is exactly 0 in all 63 values. Augmentation never turns it
  into a phantom hand (those points stay 0).
- The shoulders are the reference of the normalization (center 0, width 1), so they are
  never scaled, shifted or noised. Rotation turns the whole body.
- Speed change picks the nearest frame (no blending), so zeros are never mixed with real values.

Never augment validation or test data.
"""
import numpy as np

import config

N_POINTS = config.FEATURES_PER_FRAME // config.N_COORDS  # 48 landmarks
_SHOULDER_POINTS = [config.POSE_SLICE.start // config.N_COORDS + config.POSE_INDICES.index(i)
                    for i in (11, 12)]

# Default ranges (units: shoulder widths, because X is normalized by shoulder width)
SCALE_RANGE = (0.9, 1.1)
SHIFT_MAX = 0.1
ROTATION_MAX_DEG = 10.0
NOISE_SIGMA = 0.01
SPEED_RANGE = (0.8, 1.25)


def _points(sequence):
    """[T, 144] -> [T, 48, 3] (a copy)."""
    return sequence.reshape(len(sequence), N_POINTS, config.N_COORDS).copy()


def _movable_mask(sequence):
    """[T, 48] bool: landmarks that exist in the frame and are not shoulders."""
    mask = np.zeros((len(sequence), N_POINTS), dtype=bool)
    for block in (config.LEFT_HAND_SLICE, config.RIGHT_HAND_SLICE, config.POSE_SLICE):
        first, last = block.start // config.N_COORDS, block.stop // config.N_COORDS
        mask[:, first:last] = np.any(sequence[:, block] != 0, axis=1, keepdims=True)
    mask[:, _SHOULDER_POINTS] = False
    return mask


def scale(sequence, factor):
    """Multiply the coordinates of existing non-shoulder landmarks by `factor`."""
    points, movable = _points(sequence), _movable_mask(sequence)
    points = np.where(movable[..., None], points * factor, points)
    return points.reshape(sequence.shape).astype(np.float32)


def shift(sequence, offset):
    """Add offset (x, y, z) to existing non-shoulder landmarks."""
    points, movable = _points(sequence), _movable_mask(sequence)
    points = np.where(movable[..., None], points + np.asarray(offset, dtype=np.float32), points)
    return points.reshape(sequence.shape).astype(np.float32)


def rotate(sequence, angle_deg):
    """Rotate every landmark around the center of the shoulders, in the x-y plane."""
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    points = _points(sequence)
    x, y = points[..., 0].copy(), points[..., 1].copy()
    points[..., 0] = x * cos - y * sin
    points[..., 1] = x * sin + y * cos
    return points.reshape(sequence.shape).astype(np.float32)


def add_noise(sequence, sigma, rng):
    """Add Gaussian noise to existing non-shoulder landmarks."""
    points, movable = _points(sequence), _movable_mask(sequence)
    noise = rng.normal(0.0, sigma, size=points.shape).astype(np.float32)
    points = np.where(movable[..., None], points + noise, points)
    return points.reshape(sequence.shape).astype(np.float32)


def change_speed(sequence, factor):
    """factor > 1 plays the sign faster, < 1 slower. Nearest frame, no blending."""
    length = len(sequence)
    center = (length - 1) / 2
    positions = center + (np.arange(length) - center) * factor
    return sequence[np.clip(np.round(positions).astype(int), 0, length - 1)].copy()


def augment_sequence(sequence, rng):
    """One random augmented copy of a [30, 144] sequence (all five transformations)."""
    out = change_speed(sequence, rng.uniform(*SPEED_RANGE))
    out = rotate(out, rng.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG))
    out = scale(out, rng.uniform(*SCALE_RANGE))
    out = shift(out, rng.uniform(-SHIFT_MAX, SHIFT_MAX, size=config.N_COORDS))
    return add_noise(out, NOISE_SIGMA, rng)


def augment_batch(X, y, copies=1, seed=0):
    """Return ONLY the augmented samples: (X_aug, y_aug) with len(X) * copies rows.

    Typical use (training data only):
        X_aug, y_aug = augment_batch(X_train, y_train, copies=2, seed=42)
        X_train = np.concatenate([X_train, X_aug]); y_train = np.concatenate([y_train, y_aug])
    """
    if X.ndim != 3 or X.shape[1:] != (config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME):
        raise ValueError(f"Expected [N, {config.SEQUENCE_LENGTH}, {config.FEATURES_PER_FRAME}], got {X.shape}")
    rng = np.random.default_rng(seed)
    X_aug = np.stack([augment_sequence(sample, rng) for _ in range(copies) for sample in X])
    return X_aug, np.tile(y, copies)