# best_gru.keras: current model (for the App Engineer)

GRU, 2 layers x 128 units, dropout 0.3, learning rate 1e-3, trained on the training
participants plus 2 augmented copies (`augment.py`) and a left/right mirrored copy of all
of it (run `gru_2x128_d0.3_aug2_mirror_s2`, chosen on validation accuracy; see
`results.csv` for all runs). Input and output are the same as the previous version, so
nothing changes on the app side.

| | accuracy |
|---|---|
| validation (3 unseen participants) | 67.8% |
| test (3 unseen participants) | 88.0% |
| **cross-validation, all 21 participants tested once (same settings)** | **79.3% per person** |

The cross-validation number is the honest expectation for a new person. The official
validation group happens to contain the hardest signers and the test group some of the
easiest, which is why those two numbers are 20 points apart. Accuracy per person ranges
from about 47% to 92% (`python cv.py --summary`).

Previous version (no augmentation): 64.3% validation, 83.8% test, 74.2% cross-validation.

Expect mistakes on a new person, mostly for `go` (63% recall on test, lower on validation),
`book` and `where`, and when the hands are only visible for a few frames. Left-handed
signing is handled (the mirrored training copy covers both hands).

## Input and output
- Input: float32 `[batch, 30, 144]`, produced ONLY by `features.preprocess_sample()`.
- Output: softmax `[batch, 15]`. Index i is the word `config.WORDS[i]`.

## Using it live
```python
import numpy as np
from tensorflow import keras
import config, features

model = keras.models.load_model("models/best_gru.keras", compile=False)

# results = MediaPipe Holistic output for one camera frame
frame = features.extract_from_mediapipe(results)      # float32 [144], NaN where missing
buffer.append(frame)                                  # keep the last N frames
window = np.stack(buffer)                             # [n_frames, 144]
x = features.preprocess_sample(window)[None]          # [1, 30, 144]
probs = model.predict(x, verbose=0)[0]
word = config.WORDS[int(probs.argmax())]
```

## Things that must match training
- Give MediaPipe the raw (unmirrored) camera image. Mirror only what you display.
  A mirrored input swaps left and right hands. (The model has seen both hands, but the
  pose and hand labels must still be consistent with each other.)
- Do not normalize, augment or resample yourself: `preprocess_sample()` does it all.
  Augmentation and mirroring are training-only.
- The training samples had a median of 22 original frames (range 2-308), resampled to 30.
  The buffer length is a choice to tune on live tests; `preprocess_sample` accepts any length.
- Requires TensorFlow 2.15.1 (see `requirements.txt`).
