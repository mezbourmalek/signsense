"""Shared preprocessing for SignSense (training and live app).

Input : one sample as float32 [n_frames, 144], NaN where a landmark is missing
        (layout defined in config.py).
Output: float32 [30, 144] without NaN.

Order: normalize -> fill short gaps -> zero the rest -> resample.
"""
import numpy as np

import config

MIN_SHOULDER_WIDTH = 1e-6
# Position of the two shoulders (pose landmarks 11 and 12) among the 48 landmarks.
_SHOULDERS = [config.POSE_SLICE.start // config.N_COORDS + config.POSE_INDICES.index(i)
              for i in (11, 12)]


def normalize(sample):
    """Center each frame on the midpoint of the shoulders and divide by shoulder width.

    The width is measured in x and y only. Missing values stay NaN.
    """
    points = sample.reshape(len(sample), -1, config.N_COORDS)       # [n, 48, 3]
    shoulders = points[:, _SHOULDERS, :]                            # [n, 2, 3]
    center = shoulders.mean(axis=1, keepdims=True)                  # [n, 1, 3]
    width = np.linalg.norm(shoulders[:, 0, :2] - shoulders[:, 1, :2], axis=1)
    width = np.where(width > MIN_SHOULDER_WIDTH, width, np.nan)     # bad frame -> NaN
    normalized = (points - center) / width[:, None, None]
    return normalized.reshape(len(sample), -1).astype(np.float32)


def fill_short_gaps(sample, max_gap=config.MAX_GAP_FRAMES):
    """Linear interpolation inside gaps of at most max_gap frames, separately per hand.

    A gap is a run of missing frames with a real frame on both sides. Longer gaps,
    and frames before the first or after the last real frame, stay NaN.
    """
    out = sample.copy()
    for block in (config.LEFT_HAND_SLICE, config.RIGHT_HAND_SLICE):
        present = np.flatnonzero(~np.isnan(out[:, block]).all(axis=1))
        for before, after in zip(present[:-1], present[1:]):
            gap = after - before - 1
            if 0 < gap <= max_gap:
                weight = np.linspace(0, 1, gap + 2)[1:-1, None]
                out[before + 1:after, block] = (out[before, block] * (1 - weight)
                                                + out[after, block] * weight)
    return out


def resample(sample, length=config.SEQUENCE_LENGTH):
    """Pick `length` frames evenly spaced in time (nearest frame, no blending)."""
    if len(sample) == 0:
        raise ValueError("Cannot resample an empty sample.")
    indices = np.round(np.linspace(0, len(sample) - 1, length)).astype(int)
    return sample[indices]


def preprocess_sample(sample):
    """Full pipeline for one sample. Returns float32 [30, 144] without NaN."""
    if sample.ndim != 2 or sample.shape[1] != config.FEATURES_PER_FRAME or len(sample) == 0:
        raise ValueError(f"Expected [n_frames, {config.FEATURES_PER_FRAME}], got {sample.shape}")
    result = normalize(sample)
    result = fill_short_gaps(result)
    result = np.nan_to_num(result, nan=0.0)
    return resample(result).astype(np.float32)