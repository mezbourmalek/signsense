"""Check the landmark files of the selected words (read-only).

Writes one row per file to a CSV report outside the repository.
Nothing is modified or deleted.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

# Landmarks per frame produced by MediaPipe Holistic (as seen in the data)
EXPECTED = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}
COLUMNS = ["frame", "type", "landmark_index", "x", "y", "z"]

# Temporary copy of the candidate vocabulary; it will move to config.py
SELECTED_WORDS = [
    "hello", "thankyou", "please", "yes", "no", "wait", "finish", "where",
    "go", "home", "drink", "book", "airplane", "car", "jump",
]


def validate_file(path):
    """Return quality measures for one parquet file."""
    d = pd.read_parquet(path, columns=COLUMNS)
    types = list(EXPECTED)

    # Rows per frame and per landmark type
    counts = (
        d.groupby(["frame", "type"]).size().unstack(fill_value=0)
        .reindex(columns=types, fill_value=0)
    )
    frames = counts.index.to_numpy()

    # Rows with a real coordinate (not NaN), per frame and per type
    valid = (
        d.groupby(["frame", "type"])["x"].count().unstack(fill_value=0)
        .reindex(index=counts.index, columns=types, fill_value=0)
    )
    left = valid["left_hand"]
    right = valid["right_hand"]

    # landmark_index must go from 0 to n-1 for each type
    idx = d.groupby("type")["landmark_index"].agg(["min", "max", "nunique"])
    bad_index = False
    for t, n in EXPECTED.items():
        if t not in idx.index:
            bad_index = True
        elif (idx.loc[t, "min"] != 0 or idx.loc[t, "max"] != n - 1
              or idx.loc[t, "nunique"] != n):
            bad_index = True

    body = d[d["type"] != "face"]  # hands and pose only (face is not used)

    return {
        "n_frames": len(frames),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "frame_gaps": int(frames.max() - frames.min() + 1 - len(frames)),
        "bad_structure_frames": int((counts != pd.Series(EXPECTED)).any(axis=1).sum()),
        "bad_index": bad_index,
        "duplicated_landmarks": int(d.duplicated(["frame", "type", "landmark_index"]).sum()),
        "left_full_frames": int((left == 21).sum()),
        "left_partial_frames": int(((left > 0) & (left < 21)).sum()),
        "right_full_frames": int((right == 21).sum()),
        "right_partial_frames": int(((right > 0) & (right < 21)).sum()),
        "any_hand_frames": int(((left > 0) | (right > 0)).sum()),
        "both_hands_frames": int(((left > 0) & (right > 0)).sum()),
        "pose_incomplete_frames": int((valid["pose"] < 33).sum()),
        "x_min": body["x"].min(), "x_max": body["x"].max(),
        "y_min": body["y"].min(), "y_max": body["y"].max(),
        "z_min": body["z"].min(), "z_max": body["z"].max(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate landmark files")
    parser.add_argument("--train-csv", type=Path,
                        default=Path(r"E:\SignSense_data\raw\train.csv"))
    parser.add_argument("--landmarks-root", type=Path,
                        default=Path(r"E:\SignSense_data\raw\landmarks"))
    parser.add_argument("--out", type=Path,
                        default=Path(r"E:\SignSense_data\processed\validation_report.csv"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Check only N random files (for quick tests)")
    return parser.parse_args()


def main():
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    rows = train[train["sign"].isin(SELECTED_WORDS)]
    if args.limit:
        rows = rows.sample(n=args.limit, random_state=42)

    results = []
    start = time.time()
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        record = {
            "sequence_id": row.sequence_id,
            "participant_id": row.participant_id,
            "sign": row.sign,
            "path": row.path,
        }
        try:
            record.update(validate_file(args.landmarks_root / row.path))
            record["ok"] = True
            record["error"] = ""
        except Exception as e:  # keep going: one bad file must not stop the run
            record["ok"] = False
            record["error"] = f"{type(e).__name__}: {e}"
        results.append(record)
        if i % 500 == 0:
            print(f"{i}/{len(rows)} files | {time.time() - start:.0f} s")

    report = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)

    elapsed = time.time() - start
    print(f"Checked {len(report)} files in {elapsed:.1f} s "
          f"({elapsed / len(report):.3f} s per file)")
    print("Files that could not be read:", int((~report["ok"]).sum()))
    print("Report saved to", args.out)
    print(report["n_frames"].describe())


if __name__ == "__main__":
    main()