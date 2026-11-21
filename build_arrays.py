"""Build the arrays shared with the team: X, y, groups, labels.json.

Reads the stage-1 extraction (frames.npy + index.csv), applies features.preprocess_sample
to every sample and saves, outside the repo, in PROCESSED_DIR/arrays:

- X.npy             float32 [N, 30, 144]
- y.npy             int64   [N]  index of the word in config.WORDS
- groups.npy        int64   [N]  participant_id
- sequence_ids.npy  int64   [N]  original sequence_id (traceability)
- hand_frames.npy   int64   [N]  frames with a hand in the ORIGINAL sample (before resampling)
- labels.json       the words, in label order
- metadata.json     how the arrays were built

The split is NOT applied here: use splits/split.json and groups.npy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import features

DEFAULT_SPLIT = Path(__file__).resolve().parent / "splits" / "split.json"
SPLIT_NAMES = ("train", "validation", "test")


def count_hand_frames(sample):
    """Number of frames where at least one hand has real (non-NaN) values."""
    left = ~np.isnan(sample[:, config.LEFT_HAND_SLICE]).all(axis=1)
    right = ~np.isnan(sample[:, config.RIGHT_HAND_SLICE]).all(axis=1)
    return int((left | right).sum())


def check_extraction_layout(folder):
    """Refuse to continue if the extraction was made with a different layout than config.py."""
    with open(folder / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    expected = {
        "words": config.WORDS,
        "features_per_frame": config.FEATURES_PER_FRAME,
        "layout": {"left_hand": [config.LEFT_HAND_SLICE.start, config.LEFT_HAND_SLICE.stop],
                   "right_hand": [config.RIGHT_HAND_SLICE.start, config.RIGHT_HAND_SLICE.stop],
                   "pose": [config.POSE_SLICE.start, config.POSE_SLICE.stop]},
        "pose_indices": config.POSE_INDICES,
    }
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"metadata.json ({key}) differs from config.py. "
                             "Run extract_landmarks.py again.")
    if meta["limit"] is not None:
        raise ValueError("This extraction was a test run (--limit). Run the full extraction.")
    return meta


def split_of_each_sample(split_path, groups):
    """Return an array with 'train' / 'validation' / 'test' for every sample."""
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)["groups"]
    split_of = {p: name for name in SPLIT_NAMES for p in split[name]}
    unknown = set(groups.tolist()) - set(split_of)
    if unknown:
        raise ValueError(f"Participants missing from split.json: {sorted(unknown)}")
    return np.array([split_of[g] for g in groups.tolist()])


def verify_arrays(out_dir, X_built, index, split_labels):
    """Reload everything from disk and check it. Raise ValueError on any problem."""
    X = np.load(out_dir / "X.npy")
    y = np.load(out_dir / "y.npy")
    groups = np.load(out_dir / "groups.npy")
    hand_frames = np.load(out_dir / "hand_frames.npy")
    n = len(index)
    if X.shape != (n, config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME) or X.dtype != np.float32:
        raise ValueError(f"Unexpected X: {X.shape} {X.dtype}")
    if y.shape != (n,) or groups.shape != (n,) or hand_frames.shape != (n,):
        raise ValueError("y, groups and hand_frames must all have shape [N]")
    if not np.array_equal(X, X_built):
        raise ValueError("X read back from disk differs from X in memory")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinite values")
    if y.min() < 0 or y.max() >= len(config.WORDS):
        raise ValueError("y outside the vocabulary")
    with open(out_dir / "labels.json", encoding="utf-8") as f:
        if json.load(f)["labels"] != config.WORDS:
            raise ValueError("labels.json does not match config.WORDS")
    if not np.array_equal(groups, index["participant_id"].to_numpy()):
        raise ValueError("groups.npy does not match index.csv")
    table = pd.crosstab(pd.Series(y).map(dict(enumerate(config.WORDS))).rename("word"),
                        pd.Series(split_labels).rename("split"))[list(SPLIT_NAMES)]
    if (table == 0).any().any():
        raise ValueError(f"A word is missing from a split:\n{table}")
    return X, table


def main():
    parser = argparse.ArgumentParser(description="Build X, y, groups from the extraction.")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    args = parser.parse_args()

    t0 = time.time()
    folder = config.PROCESSED_DIR / "extracted"
    out_dir = config.PROCESSED_DIR / "arrays"
    meta = check_extraction_layout(folder)
    frames = np.load(folder / "frames.npy")
    index = pd.read_csv(folder / "index.csv")

    n = len(index)
    X = np.empty((n, config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME), dtype=np.float32)
    hand_frames = np.empty(n, dtype=np.int64)
    for i, row in enumerate(index.itertuples()):
        sample = frames[row.start:row.start + row.n_frames]
        hand_frames[i] = count_hand_frames(sample)   # measured on the raw sample
        X[i] = features.preprocess_sample(sample)

    y = index["label"].to_numpy(dtype=np.int64)
    groups = index["participant_id"].to_numpy(dtype=np.int64)
    split_labels = split_of_each_sample(args.split, groups)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "groups.npy", groups)
    np.save(out_dir / "sequence_ids.npy", index["sequence_id"].to_numpy(dtype=np.int64))
    np.save(out_dir / "hand_frames.npy", hand_frames)
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": config.WORDS,
                   "label_to_index": {w: i for i, w in enumerate(config.WORDS)}}, f, indent=2)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"n_samples": n, "sequence_length": config.SEQUENCE_LENGTH,
                   "features_per_frame": config.FEATURES_PER_FRAME,
                   "max_gap_frames": config.MAX_GAP_FRAMES,
                   "pose_indices": config.POSE_INDICES,
                   "layout": meta["layout"],
                   "samples_excluded_at_extraction": meta["n_samples_excluded"],
                   "pipeline": "normalize -> fill gaps <= max_gap_frames -> NaN=0 -> resample (nearest)"},
                  f, indent=2)

    X_disk, table = verify_arrays(out_dir, X, index, split_labels)

    hand_in_x = (X_disk[:, :, :126] != 0).any(axis=2).sum(axis=1)
    print(f"X {X_disk.shape} {X_disk.dtype}   y {y.shape}   groups {groups.shape}")
    print(f"X range: min {X_disk.min():.2f}, max {X_disk.max():.2f}; "
          f"zero values {(X_disk == 0).mean() * 100:.1f}%")
    print(f"Participants: {len(set(groups.tolist()))}")
    print("\nSamples per split:")
    print(pd.Series(split_labels).value_counts().reindex(list(SPLIT_NAMES)).to_string())
    print("\nSamples per word per split:")
    print(table.to_string())
    print("\nOriginal frames with a hand per sample (percentiles 5/25/50/75):",
          np.percentile(hand_frames, [5, 25, 50, 75]).round(1).tolist())
    print(f"Samples with a hand in the original but NO hand frame in X: "
          f"{int(((hand_frames > 0) & (hand_in_x == 0)).sum())}")
    print(f"Samples with fewer than 3 hand frames in X: {int((hand_in_x < 3).sum())}")
    print(f"\nTime: {time.time() - t0:.0f}s   Output: {out_dir}")
    print("Verification passed.")


if __name__ == "__main__":
    main()