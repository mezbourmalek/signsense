"""Train a GRU or LSTM on the prepared SignSense arrays.

Example:
    python train.py --arrays-dir path\\to\\arrays --model gru --units 128 --name gru_128

Every run is appended to models/results.csv (settings, validation and test accuracy)
and the best model (by validation accuracy) is saved to models/<name>.keras.
Model selection uses the validation set only. Test accuracy is recorded, not tuned on.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

import config
from load_data import DEFAULT_SPLIT, load_split

MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULT_FIELDS = ["name", "model", "layers", "units", "dropout", "lr", "batch_size",
                 "epochs_run", "min_hand_frames_train", "seed", "val_acc", "test_acc",
                 "train_seconds", "n_params", "model_file", "augment_copies", "mirror", "velocity", "split"]
# Value written for older log rows that lack a newer column.
COLUMN_DEFAULTS = {"augment_copies": "0", "mirror": "0", "velocity": "0", "split": "split.json"}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def mirror(X):
    """Left/right mirror of preprocessed sequences [N, 30, 144].

    A sign made with the left hand becomes the same sign made with the right hand:
    x is negated (X is centered on the shoulders, so this flips around the body
    midline), the two hand blocks swap places, and the left/right pose points
    swap (shoulder 11<->12, elbow 13<->14, wrist 15<->16). Missing hands stay 0.
    """
    pts = X.reshape(len(X), X.shape[1], -1, config.N_COORDS).copy()
    pts[..., 0] = -pts[..., 0]
    n_hand = config.HAND_LANDMARKS
    order = list(range(n_hand, 2 * n_hand)) + list(range(n_hand))
    pose_first = 2 * n_hand
    swapped = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
    order += [pose_first + config.POSE_INDICES.index(swapped[i]) for i in config.POSE_INDICES]
    out = pts[:, :, order].reshape(X.shape)
    return out + 0.0  # turns -0.0 into 0.0 so missing hands stay exactly 0


def build_model(kind, layers, units, dropout, lr, velocity=False):
    from tensorflow import keras
    from tensorflow.keras import layers as L
    cell = {"gru": L.GRU, "lstm": L.LSTM}[kind]
    inputs = keras.Input(shape=(config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME))
    x = inputs
    if velocity:
        # Frame-to-frame change, computed inside the model so the input stays [30, 144]
        # (only built-in layers, so the saved .keras file loads without custom code).
        diff = L.Subtract()([L.Cropping1D((1, 0))(inputs), L.Cropping1D((0, 1))(inputs)])
        x = L.Concatenate()([inputs, L.ZeroPadding1D((1, 0))(diff)])
    for i in range(layers):
        x = cell(units, return_sequences=i < layers - 1, dropout=dropout)(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(64, activation="relu")(x)
    outputs = L.Dense(len(config.WORDS), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"{kind}_{layers}x{units}")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def log_result(row, path=None):
    MODELS_DIR.mkdir(exist_ok=True)
    path = Path(path) if path else MODELS_DIR / "results.csv"
    new = not path.exists()
    if not new:
        # Older logs lack newer columns: rewrite them once with the full header.
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields, old_rows = reader.fieldnames, list(reader)
        if old_fields != RESULT_FIELDS:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                for r in old_rows:
                    writer.writerow({k: r.get(k) or COLUMN_DEFAULTS.get(k, "")
                                     for k in RESULT_FIELDS})
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Train a GRU/LSTM on SignSense.")
    p.add_argument("--arrays-dir", default=os.environ.get("SIGNSENSE_ARRAYS"),
                   help="folder with X.npy, y.npy, groups.npy, hand_frames.npy "
                        "(or set SIGNSENSE_ARRAYS)")
    p.add_argument("--model", choices=["gru", "lstm"], default="gru")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-hand-frames-train", type=int, default=0)
    p.add_argument("--augment", type=int, default=0,
                   help="add N augmented copies of the TRAINING set (augment.py); "
                        "validation and test are never augmented")
    p.add_argument("--mirror", action="store_true",
                   help="also train on a left/right mirrored copy of the (augmented) "
                        "training set, so each sign is seen with both hands")
    p.add_argument("--velocity", action="store_true",
                   help="add frame-to-frame differences inside the model")
    p.add_argument("--split", default=str(DEFAULT_SPLIT),
                   help="split file (default splits/split.json; cv.py passes fold files)")
    p.add_argument("--out-dir", default=str(MODELS_DIR),
                   help="where the .keras/.json files go")
    p.add_argument("--results", default=None,
                   help="results CSV (default models/results.csv)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if not args.arrays_dir:
        raise SystemExit("Give --arrays-dir or set SIGNSENSE_ARRAYS.")

    from tensorflow import keras
    set_seed(args.seed)
    name = args.name or f"{args.model}_{args.layers}x{args.units}_d{args.dropout}_lr{args.lr}"

    data = load_split(args.arrays_dir, split_path=args.split,
                      min_hand_frames_train=args.min_hand_frames_train, with_groups=True)
    (X_train, y_train, _), (X_val, y_val, g_val), (X_test, y_test, g_test) = (
        data["train"], data["validation"], data["test"])
    if args.augment > 0:
        import augment
        X_aug, y_aug = augment.augment_batch(X_train, y_train, copies=args.augment,
                                             seed=args.seed)
        X_train = np.concatenate([X_train, X_aug])
        y_train = np.concatenate([y_train, y_aug])
    if args.mirror:
        X_train = np.concatenate([X_train, mirror(X_train)])
        y_train = np.concatenate([y_train, y_train])
    print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    model = build_model(args.model, args.layers, args.units, args.dropout, args.lr,
                        velocity=args.velocity)
    model.summary()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / f"{name}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(model_file), monitor="val_accuracy",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(3, args.patience // 3)),
    ]
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=callbacks, verbose=2)
    seconds = time.time() - t0

    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    per_participant = {}
    for split, X_s, y_s, g_s in (("validation", X_val, y_val, g_val),
                                 ("test", X_test, y_test, g_test)):
        pred = model.predict(X_s, verbose=0).argmax(1)
        per_participant[split] = {int(p): {"n": int((g_s == p).sum()),
                                           "acc": round(float((pred[g_s == p] == y_s[g_s == p]).mean()), 4)}
                                  for p in np.unique(g_s)}
    print(f"\nvalidation accuracy: {val_acc:.4f}   test accuracy: {test_acc:.4f}")

    log_result({
        "name": name, "model": args.model, "layers": args.layers, "units": args.units,
        "dropout": args.dropout, "lr": args.lr, "batch_size": args.batch_size,
        "epochs_run": len(history.history["loss"]),
        "min_hand_frames_train": args.min_hand_frames_train, "seed": args.seed,
        "val_acc": round(float(val_acc), 4), "test_acc": round(float(test_acc), 4),
        "train_seconds": round(seconds), "n_params": model.count_params(),
        "model_file": model_file.name, "augment_copies": args.augment,
        "mirror": int(args.mirror), "velocity": int(args.velocity),
        "split": Path(args.split).name,
    }, path=args.results)
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"words": config.WORDS, "input_shape": [config.SEQUENCE_LENGTH,
                                                          config.FEATURES_PER_FRAME],
                   "args": vars(args), "per_participant": per_participant}, f, indent=2)
    print(f"Saved {model_file} and logged the run in {args.results or MODELS_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
