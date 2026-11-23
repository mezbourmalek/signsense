"""Run a small grid of train.py experiments, one after another.

    python sweep.py --arrays-dir path\\to\\arrays

Grid: GRU and LSTM x units {64, 128, 256} x dropout {0.3, 0.5}, all with 2 layers and
lr 1e-3. Then two extra runs (GRU and LSTM, 128 units, dropout 0.3) that drop training
samples with fewer than 5 hand frames. Runs already present in models/results.csv are
skipped, so the script can be stopped and restarted. Results go to models/results.csv.
"""
import argparse
import csv
import itertools
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "models" / "results.csv"


def done_names():
    if not RESULTS.exists():
        return set()
    with open(RESULTS, newline="", encoding="utf-8") as f:
        return {row["name"] for row in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrays-dir", default=os.environ.get("SIGNSENSE_ARRAYS"))
    p.add_argument("--stage", choices=["size", "lr"], default="size",
                   help="size: model/units/dropout grid. lr: learning rates on the "
                        "128-unit GRU, dropout 0.3, three seeds each")
    args = p.parse_args()
    if not args.arrays_dir:
        raise SystemExit("Give --arrays-dir or set SIGNSENSE_ARRAYS.")

    runs = []
    if args.stage == "size":
        for kind, units, dropout in itertools.product(("gru", "lstm"), (64, 128, 256),
                                                      (0.3, 0.5)):
            runs.append((f"{kind}_2x{units}_d{dropout}",
                         ["--model", kind, "--units", str(units), "--dropout", str(dropout)]))
        for kind in ("gru", "lstm"):
            runs.append((f"{kind}_2x128_d0.3_hand5",
                         ["--model", kind, "--units", "128", "--dropout", "0.3",
                          "--min-hand-frames-train", "5"]))
    else:
        # Three seeds per learning rate: one run per setting is too noisy to compare.
        for lr, seed in itertools.product((3e-4, 1e-3, 3e-3), (42, 1, 2)):
            name = "gru_2x128_d0.3" if (lr, seed) == (1e-3, 42) else f"gru_2x128_d0.3_lr{lr}_s{seed}"
            runs.append((name, ["--model", "gru", "--units", "128", "--dropout", "0.3",
                                "--lr", str(lr), "--seed", str(seed)]))

    finished = done_names()
    for i, (name, extra) in enumerate(runs, start=1):
        if name in finished:
            print(f"[{i}/{len(runs)}] {name}: already done, skipped", flush=True)
            continue
        print(f"[{i}/{len(runs)}] {name}", flush=True)
        cmd = [sys.executable, str(HERE / "train.py"), "--arrays-dir", args.arrays_dir,
               "--name", name] + extra
        subprocess.run(cmd, check=False, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Sweep finished. See models/results.csv")


if __name__ == "__main__":
    main()
