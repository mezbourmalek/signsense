"""
This script performs an exhaustive participant-based split into three subsets:

- Training: 15 participants (approximately 71.4%)
- Validation: 3 participants (approximately 14.3%)
- Test: 3 participants (approximately 14.3%)

It evaluates all valid split combinations and selects the best one according to the defined balancing criteria.

The split is performed at the participant level to prevent data leakage between training, validation, and test sets.

The selected split is saved in JSON format, along with a CSV report containing participant assignments and their statistics.

The script also generates diagnostic information to help verify the distribution of samples across the three sets.

"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

WORDS = ["hello", "thankyou", "please", "yes", "no", "wait", "finish", "where",
         "go", "home", "drink", "book", "airplane", "car", "jump"]
REQUIRED_COLUMNS = ["participant_id", "samples", "median_frames",
                    "pct_under_10", "dominant_hand"]
VALID_HANDS = {"left", "right", "mixed"}
EXPECTED_SIZES = {"train": 15, "validation": 3, "test": 3}
SIGMA_MIN = 1e-9  # below this, a score term carries no information

DEFAULT_SUMMARY = r"E:\SignSense_data\processed\participant_summary.csv"
DEFAULT_TRAIN_CSV = r"E:\SignSense_data\raw\train.csv"
DEFAULT_OUT_DIR = r"E:\SignSense_data\processed"


def load_summary(path):
    """Read participant_summary.csv and validate it. Raise ValueError if anything is wrong."""
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {path}: {missing_cols}")
    df = df[REQUIRED_COLUMNS].copy()
    if df.isna().any().any():
        raise ValueError("The summary contains empty cells.")
    if df["participant_id"].duplicated().any():
        raise ValueError("Duplicate participant_id in the summary.")
    if (df["samples"] <= 0).any() or (df["median_frames"] <= 0).any():
        raise ValueError("samples and median_frames must be positive.")
    if ((df["pct_under_10"] < 0) | (df["pct_under_10"] > 100)).any():
        raise ValueError("pct_under_10 must be between 0 and 100.")
    bad_hands = set(df["dominant_hand"]) - VALID_HANDS
    if bad_hands:
        raise ValueError(f"Unexpected dominant_hand values: {bad_hands}")
    df["participant_id"] = df["participant_id"].astype(int)
    if (df["dominant_hand"] == "left").sum() < 2 or (df["dominant_hand"] == "right").sum() < 4:
        raise ValueError("Need at least 2 left and 4 right participants.")
    if len(df) != sum(EXPECTED_SIZES.values()):
        raise ValueError(f"Expected 21 participants, found {len(df)}.")
    return df


def file_hash(path):
    """SHA-256 of the file after normalizing Windows/Linux line endings."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def enumerate_splits(left_ids, right_ids):
    """All valid (validation_ids, test_ids) pairs.

    For the two left participants, the smaller ID goes to validation. Otherwise
    every split would appear twice (validation and test swapped).
    """
    splits = []
    for left_val, left_test in combinations(left_ids, 2):
        for right_val in combinations(right_ids, 2):
            rest = [p for p in right_ids if p not in right_val]
            for right_test in combinations(rest, 2):
                splits.append(((left_val,) + right_val, (left_test,) + right_test))
    return splits


def group_features(ids, info):
    """x1 = total samples, x2 = share of samples under 10 frames, x3 = mean of medians."""
    n_total = sum(info[p]["n"] for p in ids)
    under = sum(info[p]["n"] * info[p]["u"] for p in ids)
    return [n_total, under / n_total, sum(info[p]["m"] for p in ids) / len(ids)]


def find_best_split(df):
    """Exhaustive search. Returns the best split and the numbers used to choose it."""
    info = {int(r.participant_id): {"n": float(r.samples), "m": float(r.median_frames),
                                    "u": float(r.pct_under_10) / 100.0}
            for r in df.itertuples()}
    left_ids = sorted(df.loc[df["dominant_hand"] == "left", "participant_id"])
    right_ids = sorted(df.loc[df["dominant_hand"] == "right", "participant_id"])
    mixed_ids = sorted(df.loc[df["dominant_hand"] == "mixed", "participant_id"])

    # Targets are computed over all 21 participants (not only eligible ones).
    # The difference is negligible (about 0.06% in size) and it keeps the target simple.
    all_ids = sorted(info)
    total_n = sum(info[p]["n"] for p in all_ids)
    targets = np.array([
        total_n * 3 / 21,
        sum(info[p]["n"] * info[p]["u"] for p in all_ids) / total_n,
        sum(info[p]["m"] for p in all_ids) / len(all_ids),
    ])

    splits = enumerate_splits(left_ids, right_ids)
    x_val = np.array([group_features(v, info) for v, _ in splits])
    x_test = np.array([group_features(t, info) for _, t in splits])

    # Standard deviation of each feature over ALL candidate groups (ddof=0).
    sigma = np.concatenate([x_val, x_test]).std(axis=0)
    warnings = []
    weights = np.ones(3)
    names = ["size", "share_under_10", "mean_median_frames"]
    for k in range(3):
        if sigma[k] < SIGMA_MIN:
            weights[k] = 0.0
            sigma[k] = 1.0
            warnings.append(f"Score term '{names[k]}' dropped: sigma is ~0.")

    s_val = (np.abs(x_val - targets) / sigma * weights).sum(axis=1)
    s_test = (np.abs(x_test - targets) / sigma * weights).sum(axis=1)
    worst = np.round(np.maximum(s_val, s_test), 12)
    total = np.round(s_val + s_test, 12)
    # Lexicographic order: worst group first, then the sum. Ties: enumeration
    # order, which is fixed because the IDs are sorted.
    best = int(np.lexsort((np.arange(len(splits)), total, worst))[0])

    val_ids, test_ids = splits[best]
    train_ids = sorted(set(all_ids) - set(val_ids) - set(test_ids))
    if not set(mixed_ids) <= set(train_ids):
        raise ValueError("Internal error: a mixed participant is not in train.")
    details = {
        "n_splits_evaluated": len(splits),
        "targets": dict(zip(names, map(float, targets))),
        "sigma": dict(zip(names, map(float, sigma))),
        "score": {"worst_group": float(worst[best]), "sum": float(total[best])},
        "warnings": warnings,
    }
    return {"train": [int(p) for p in train_ids],
            "validation": sorted(int(p) for p in val_ids),
            "test": sorted(int(p) for p in test_ids)}, details


def verify_saved_split(split_path, df):
    """Re-read split.json from disk and check every rule. Raise ValueError on failure."""
    with open(split_path, encoding="utf-8") as f:
        groups = json.load(f)["groups"]
    hand = dict(zip(df["participant_id"].tolist(), df["dominant_hand"].tolist()))
    sizes = {name: len(groups[name]) for name in EXPECTED_SIZES}
    if sizes != EXPECTED_SIZES:
        raise ValueError(f"Wrong group sizes: {sizes}")
    all_ids = [p for name in EXPECTED_SIZES for p in groups[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A participant appears in more than one group.")
    if set(all_ids) != set(hand):
        raise ValueError("The saved split does not cover exactly the 21 participants.")
    for name in ("validation", "test"):
        if sorted(hand[p] for p in groups[name]) != ["left", "right", "right"]:
            raise ValueError(f"{name} must contain 1 left + 2 right participants.")
    if any(hand[p] == "mixed" for p in groups["validation"] + groups["test"]):
        raise ValueError("The mixed participant must stay in train.")


def main():
    parser = argparse.ArgumentParser(description="Create the participant-based split.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = load_summary(args.summary)
    groups, details = find_best_split(df)

    group_of = {p: name for name, ids in groups.items() for p in ids}
    report = df.copy()
    report["group"] = report["participant_id"].map(group_of)
    report = report.sort_values(["group", "participant_id"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "split.json"
    result = {
        "rules": "15/3/3; validation and test = 1 left + 2 right; mixed participant in train",
        "input_file": Path(args.summary).name,  # file name only, no local path
        "input_sha256_normalized": file_hash(args.summary),
        **details,
        "groups": groups,
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    report.to_csv(out_dir / "split_report.csv", index=False)
    verify_saved_split(split_path, df)

    print(f"Splits evaluated: {details['n_splits_evaluated']}")
    print(f"Score (worst group, sum): {details['score']['worst_group']:.3f}, "
          f"{details['score']['sum']:.3f}")
    for w in details["warnings"]:
        print("WARNING:", w)
    summary = report.groupby("group").agg(
        participants=("participant_id", "count"), samples=("samples", "sum"),
        mean_median_frames=("median_frames", "mean"),
        mean_pct_under_10=("pct_under_10", "mean"))
    print(summary.round(1).to_string())

    # Diagnostic only: samples per word per group (not part of the score).
    train = pd.read_csv(args.train_csv)
    train = train[train["sign"].isin(WORDS)].copy()
    train["group"] = train["participant_id"].map(group_of)
    if train["group"].isna().any():
        raise ValueError("train.csv has participants missing from the split.")
    counts = pd.crosstab(train["sign"], train["group"])[["train", "validation", "test"]]
    print("\nSamples per word per group:")
    print(counts.to_string())
    print("\nVerification passed: 15/3/3, 1 left + 2 right in validation and test, "
          "mixed in train, all 21 participants covered.")


if __name__ == "__main__":
    main()