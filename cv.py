"""Cross-validation by participant: every one of the 21 participants is tested once.

    python cv.py --arrays-dir path\\to\\arrays --configs base aug_mirror aug_mirror_vel
    python cv.py --summary          # print the table again from models/cv/cv_results.csv

The 21 participants form 7 folds of 3. Fold 0 is the official test group and fold 1 the
official validation group (splits/split.json), so the run of fold 0 IS the official split.
For fold k: test = fold k, validation = fold k+1 (early stopping / checkpoint only),
train = the other 15 participants. Same 15/3/3 shape as the official split.
Each other fold has at least one left-handed signer, like the official split.

Fold files go to splits/cv/, models and results to models/cv/ (results.csv is untouched).
Runs already in models/cv/cv_results.csv are skipped, so it can be stopped and restarted.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "splits" / "split.json"
FOLD_DIR = HERE / "splits" / "cv"
OUT_DIR = HERE / "models" / "cv"
RESULTS = OUT_DIR / "cv_results.csv"

# Folds 2-6: the 15 official training participants, one left-handed signer first in each.
# (left-handed: 16069 27610 32319 34503 36257 55372; 37055 uses both hands)
OTHER_FOLDS = [
    [16069, 2044, 4718],
    [27610, 18796, 28656],
    [32319, 37779, 49445],
    [34503, 53618, 62590],
    [36257, 55372, 37055],
]

BASE = ["--model", "gru", "--units", "128", "--dropout", "0.3", "--lr", "1e-3"]
CONFIGS = {
    "base": BASE,
    "aug_mirror": BASE + ["--augment", "2", "--mirror"],
    "aug_mirror_vel": BASE + ["--augment", "2", "--mirror", "--velocity"],
    "aug_mirror_256": ["--model", "gru", "--units", "256", "--dropout", "0.3", "--lr", "1e-3",
                       "--augment", "2", "--mirror"],
}


def make_folds():
    with open(SPLIT, encoding="utf-8") as f:
        official = json.load(f)["groups"]
    folds = [official["test"], official["validation"]] + OTHER_FOLDS
    everyone = [p for name in ("train", "validation", "test") for p in official[name]]
    flat = [p for fold in folds for p in fold]
    assert sorted(flat) == sorted(everyone), "folds must cover each participant exactly once"
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(len(folds)):
        val = (k + 1) % len(folds)
        groups = {"train": [p for i, f in enumerate(folds) if i not in (k, val) for p in f],
                  "validation": folds[val], "test": folds[k]}
        path = FOLD_DIR / f"fold{k}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rules": f"cv fold {k} of {len(folds)}: test = fold {k}, "
                                f"validation = fold {val}", "groups": groups}, f, indent=2)
        paths.append(path)
    return paths


def done_names():
    if not RESULTS.exists():
        return set()
    with open(RESULTS, newline="", encoding="utf-8") as f:
        return {row["name"] for row in csv.DictReader(f)}


def summary(configs=None):
    """Per-config mean over the 21 participants (each tested once) and per participant."""
    table = {}
    for js in OUT_DIR.glob("cv_*_f*.json"):
        cfg, fold = js.stem[3:].rsplit("_f", 1)
        if configs and cfg not in configs:
            continue
        info = json.loads(js.read_text(encoding="utf-8"))
        for p, r in info["per_participant"]["test"].items():
            table.setdefault(cfg, {})[int(p)] = r
    if not table:
        print("No finished CV runs yet.")
        return
    people = sorted({p for t in table.values() for p in t})
    names = sorted(table, key=lambda c: list(CONFIGS).index(c) if c in CONFIGS else 99)
    print(f"{'participant':>11}" + "".join(f"{c:>16}" for c in names))
    for p in people:
        print(f"{p:>11}" + "".join(f"{table[c][p]['acc']:16.3f}" if p in table[c] else f"{'-':>16}"
                                   for c in names))
    print("-" * (11 + 16 * len(names)))
    for label, fn in (("mean person", lambda t: np.mean([r["acc"] for r in t.values()])),
                      ("all samples", lambda t: sum(r["acc"] * r["n"] for r in t.values())
                       / sum(r["n"] for r in t.values())),
                      ("worst person", lambda t: min(r["acc"] for r in t.values())),
                      ("people done", lambda t: len(t))):
        print(f"{label:>11}" + "".join(f"{fn(table[c]):16.3f}" if label != "people done"
                                       else f"{fn(table[c]):16d}" for c in names))
    if len(names) > 1:
        ref = names[0]
        for c in names[1:]:
            common = [p for p in table[ref] if p in table[c]]
            d = np.array([table[c][p]["acc"] - table[ref][p]["acc"] for p in common])
            print(f"{c} vs {ref}: mean {d.mean():+.3f} per person, better for "
                  f"{(d > 0).sum()}/{len(d)} people")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrays-dir", default=os.environ.get("SIGNSENSE_ARRAYS"))
    p.add_argument("--configs", nargs="+", default=["base", "aug_mirror", "aug_mirror_vel"],
                   choices=list(CONFIGS))
    p.add_argument("--folds", nargs="+", type=int, default=None, help="only these folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary", action="store_true", help="only print the table")
    args = p.parse_args()
    if args.summary:
        summary()
        return
    if not args.arrays_dir:
        raise SystemExit("Give --arrays-dir or set SIGNSENSE_ARRAYS.")

    fold_paths = make_folds()
    folds = args.folds if args.folds is not None else range(len(fold_paths))
    finished = done_names()
    for cfg in args.configs:
        for k in folds:
            name = f"cv_{cfg}_f{k}"
            if name in finished:
                print(f"{name}: already done, skipped", flush=True)
                continue
            print(name, flush=True)
            cmd = [sys.executable, str(HERE / "train.py"), "--arrays-dir", args.arrays_dir,
                   "--name", name, "--split", str(fold_paths[k]), "--out-dir", str(OUT_DIR),
                   "--results", str(RESULTS), "--seed", str(args.seed)] + CONFIGS[cfg]
            subprocess.run(cmd, check=False, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary(args.configs)


if __name__ == "__main__":
    main()
