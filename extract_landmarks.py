"""Stage 1 extraction: parquet landmark files -> one array of 144 values per frame.

No resampling, no normalization, no NaN filling here. Each sample keeps its original
length and its missing values. Those decisions belong to features.py.

Outputs (outside the repo, in PROCESSED_DIR/extracted):
- frames.npy     float32 [total_frames, 144], all samples stacked one after another
- index.csv      one row per kept sample: sequence_id, participant_id, sign, label,
                 start, n_frames  (sample i = frames[start : start + n_frames])
- excluded.csv   samples left out, with the reason
- metadata.json  the layout used (order of the 144 values, pose indices, words)


"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config

COLUMNS = ["frame", "type", "landmark_index", "x", "y", "z"]


class FrameGapError(ValueError):
    """The frame numbers of a sample are not consecutive."""


def fill_block(out, rows, first_cols, sub):
    """Write x, y, z of the rows in `sub` into out, at columns first_cols, +1, +2."""
    for k, axis in enumerate(("x", "y", "z")):
        out[rows, first_cols + k] = sub[axis].to_numpy(dtype=np.float32)


def extract_sample(path):
    """Return a float32 array [n_frames, 144] with NaN where a landmark is missing."""
    df = pd.read_parquet(path, columns=COLUMNS)
    frame_ids = np.sort(df["frame"].unique()).astype(np.int64)
    first = int(frame_ids[0])
    if not np.array_equal(frame_ids, np.arange(first, first + len(frame_ids))):
        raise FrameGapError("frame numbers are not consecutive")

    out = np.full((len(frame_ids), config.FEATURES_PER_FRAME), np.nan, dtype=np.float32)
    # int64 everywhere: parquet may store these columns as small integer types
    rows_all = df["frame"].to_numpy().astype(np.int64) - first
    types = df["type"].to_numpy()
    landmark = df["landmark_index"].to_numpy().astype(np.int64)

    for type_name, block in (("left_hand", config.LEFT_HAND_SLICE),
                             ("right_hand", config.RIGHT_HAND_SLICE)):
        mask = types == type_name
        slots = landmark[mask]
        if slots.size and (slots.min() < 0 or slots.max() >= config.HAND_LANDMARKS):
            raise ValueError(f"bad hand landmark_index in {path}")
        fill_block(out, rows_all[mask], block.start + slots * config.N_COORDS, df[mask])

    slot_of = {idx: i for i, idx in enumerate(config.POSE_INDICES)}
    mask = (types == "pose") & np.isin(landmark, config.POSE_INDICES)
    slots = np.array([slot_of[i] for i in landmark[mask]], dtype=np.int64)
    fill_block(out, rows_all[mask], config.POSE_SLICE.start + slots * config.N_COORDS, df[mask])
    return out


def verify_output(out_dir, expected_kept):
    """Reload the saved files and check them. Raise ValueError on any problem."""
    frames = np.load(out_dir / "frames.npy")
    index = pd.read_csv(out_dir / "index.csv")
    if frames.dtype != np.float32 or frames.ndim != 2 or frames.shape[1] != config.FEATURES_PER_FRAME:
        raise ValueError(f"Unexpected frames array: {frames.dtype} {frames.shape}")
    if len(index) != expected_kept or index["sequence_id"].duplicated().any():
        raise ValueError("index.csv has the wrong number of rows or duplicate sequence_ids")
    if frames.shape[0] != index["n_frames"].sum():
        raise ValueError("Total frames in frames.npy does not match index.csv")
    ends = index["start"] + index["n_frames"]
    if index["start"].iloc[0] != 0 or not (index["start"].iloc[1:].to_numpy() == ends.iloc[:-1].to_numpy()).all():
        raise ValueError("start/n_frames offsets are not contiguous")
    if not index["label"].between(0, len(config.WORDS) - 1).all():
        raise ValueError("label outside the vocabulary")
    if not (index["sign"] == index["label"].map(dict(enumerate(config.WORDS)))).all():
        raise ValueError("sign and label do not match")
    if np.isinf(frames).any():
        raise ValueError("frames.npy contains infinite values")
    return frames


def print_summary(frames, n_kept, n_excluded, elapsed, out_dir):
    """Print counts and hand-presence statistics (to compare with earlier measurements)."""
    left_block = np.isnan(frames[:, config.LEFT_HAND_SLICE])
    right_block = np.isnan(frames[:, config.RIGHT_HAND_SLICE])
    left = ~left_block.all(axis=1)
    right = ~right_block.all(axis=1)
    partial = int((left_block.any(axis=1) & left).sum() + (right_block.any(axis=1) & right).sum())
    print(f"\nSamples kept: {n_kept}, excluded: {n_excluded}")
    print(f"Frames: {len(frames)}   array: {frames.shape} {frames.dtype}")
    print(f"Both hands:  {(left & right).mean() * 100:.1f}%")
    print(f"Left only:   {(left & ~right).mean() * 100:.1f}%")
    print(f"Right only:  {(~left & right).mean() * 100:.1f}%")
    print(f"No hand:     {(~left & ~right).mean() * 100:.1f}%")
    print(f"Frames with a partially missing hand: {partial}")
    print(f"NaN values in pose block: {int(np.isnan(frames[:, config.POSE_SLICE]).sum())}")
    print(f"Time: {elapsed:.0f}s   Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 landmark extraction.")
    parser.add_argument("--limit", type=int, default=None,
                        help="extract only N random samples (seed 42), for testing")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    default_dir = config.PROCESSED_DIR / ("extracted_test" if args.limit else "extracted")
    out_dir = Path(args.out_dir) if args.out_dir else default_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(config.TRAIN_CSV, usecols=["sequence_id", "participant_id", "sign"])
    train = train[train["sign"].isin(config.WORDS)]
    if args.limit:
        train = train.sample(n=args.limit, random_state=42)
    train = train.sort_values("sequence_id").reset_index(drop=True)

    paths = {int(p.stem): p for p in config.LANDMARKS_DIR.rglob("*.parquet")}
    if not paths:
        raise SystemExit(f"No parquet files found in {config.LANDMARKS_DIR}. Check config.py.")

    arrays, kept, excluded = [], [], []
    start = 0
    t0 = time.time()
    for i, row in enumerate(train.itertuples(), start=1):
        path = paths.get(int(row.sequence_id))
        if path is None:
            excluded.append((row.sequence_id, "", "parquet file not found"))
            continue
        try:
            sample = extract_sample(path)
        except FrameGapError as err:
            excluded.append((row.sequence_id, str(path), str(err)))
            continue
        arrays.append(sample)
        kept.append((row.sequence_id, row.participant_id, row.sign,
                     config.WORDS.index(row.sign), start, len(sample)))
        start += len(sample)
        if i % 500 == 0:
            print(f"{i}/{len(train)} files, {time.time() - t0:.0f}s")

    if not arrays:
        raise SystemExit("No samples were extracted. Check TRAIN_CSV and LANDMARKS_DIR in config.py.")

    np.save(out_dir / "frames.npy", np.concatenate(arrays))
    pd.DataFrame(kept, columns=["sequence_id", "participant_id", "sign", "label",
                                "start", "n_frames"]).to_csv(out_dir / "index.csv", index=False)
    pd.DataFrame(excluded, columns=["sequence_id", "path", "reason"]).to_csv(
        out_dir / "excluded.csv", index=False)
    metadata = {
        "words": config.WORDS,
        "features_per_frame": config.FEATURES_PER_FRAME,
        "layout": {"left_hand": [config.LEFT_HAND_SLICE.start, config.LEFT_HAND_SLICE.stop],
                   "right_hand": [config.RIGHT_HAND_SLICE.start, config.RIGHT_HAND_SLICE.stop],
                   "pose": [config.POSE_SLICE.start, config.POSE_SLICE.stop]},
        "pose_indices": config.POSE_INDICES,
        "n_samples_kept": len(kept),
        "n_samples_excluded": len(excluded),
        "limit": args.limit,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    frames = verify_output(out_dir, expected_kept=len(kept))
    print_summary(frames, len(kept), len(excluded), time.time() - t0, out_dir)
    print("Verification passed.")


if __name__ == "__main__":
    main()