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
from load_data import load_split

MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULT_FIELDS = ["name", "model", "layers", "units", "dropout", "lr", "batch_size",
                 "epochs_run", "min_hand_frames_train", "seed", "val_acc", "test_acc",
                 "train_seconds", "n_params", "model_file"]


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def build_model(kind, layers, units, dropout, lr):
    from tensorflow import keras
    from tensorflow.keras import layers as L
    cell = {"gru": L.GRU, "lstm": L.LSTM}[kind]
    inputs = keras.Input(shape=(config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME))
    x = inputs
    for i in range(layers):
        x = cell(units, return_sequences=i < layers - 1, dropout=dropout)(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(64, activation="relu")(x)
    outputs = L.Dense(len(config.WORDS), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=f"{kind}_{layers}x{units}")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def log_result(row):
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "results.csv"
    new = not path.exists()
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if not args.arrays_dir:
        raise SystemExit("Give --arrays-dir or set SIGNSENSE_ARRAYS.")

    from tensorflow import keras
    set_seed(args.seed)
    name = args.name or f"{args.model}_{args.layers}x{args.units}_d{args.dropout}_lr{args.lr}"

    data = load_split(args.arrays_dir, min_hand_frames_train=args.min_hand_frames_train)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (
        data["train"], data["validation"], data["test"])
    print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    model = build_model(args.model, args.layers, args.units, args.dropout, args.lr)
    model.summary()

    MODELS_DIR.mkdir(exist_ok=True)
    model_file = MODELS_DIR / f"{name}.keras"
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
    print(f"\nvalidation accuracy: {val_acc:.4f}   test accuracy: {test_acc:.4f}")

    log_result({
        "name": name, "model": args.model, "layers": args.layers, "units": args.units,
        "dropout": args.dropout, "lr": args.lr, "batch_size": args.batch_size,
        "epochs_run": len(history.history["loss"]),
        "min_hand_frames_train": args.min_hand_frames_train, "seed": args.seed,
        "val_acc": round(float(val_acc), 4), "test_acc": round(float(test_acc), 4),
        "train_seconds": round(seconds), "n_params": model.count_params(),
        "model_file": model_file.name,
    })
    with open(MODELS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"words": config.WORDS, "input_shape": [config.SEQUENCE_LENGTH,
                                                          config.FEATURES_PER_FRAME],
                   "args": vars(args)}, f, indent=2)
    print(f"Saved {model_file} and logged the run in {MODELS_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
