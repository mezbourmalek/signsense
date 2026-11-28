"""Load the SignSense arrays and apply the participant-based split.

Usage:
    from load_data import load_split
    data = load_split(r"path\to\arrays")
    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]
    X_test, y_test = data["test"]

The split comes from splits/split.json (by participant, never by sample).
"""
import json
from pathlib import Path

import numpy as np

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT = Path(__file__).resolve().parent / "splits" / "split.json"


def load_split(arrays_dir, split_path=DEFAULT_SPLIT, min_hand_frames_train=0, with_groups=False):
    """Return {"train": (X, y), "validation": (X, y), "test": (X, y)}.

    min_hand_frames_train: drop TRAIN samples whose original sample had fewer frames
    with a hand than this (0 = keep everything). Validation and test are never filtered.
    with_groups: also return the participant ids as a third element.
    """
    arrays_dir = Path(arrays_dir)
    X = np.load(arrays_dir / "X.npy")
    y = np.load(arrays_dir / "y.npy")
    groups = np.load(arrays_dir / "groups.npy")
    hand_frames = np.load(arrays_dir / "hand_frames.npy")

    with open(split_path, encoding="utf-8") as f:
        ids = json.load(f)["groups"]
    all_ids = [p for name in SPLIT_NAMES for p in ids[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A participant appears in more than one split in split.json.")

    covered = np.zeros(len(groups), dtype=bool)
    result = {}
    for name in SPLIT_NAMES:
        mask = np.isin(groups, ids[name])
        covered |= mask
        if name == "train":
            mask &= hand_frames >= min_hand_frames_train
        result[name] = (X[mask], y[mask], groups[mask]) if with_groups else (X[mask], y[mask])
    if not covered.all():
        raise ValueError("Some samples belong to participants missing from split.json.")
    return result
