# best_gru.keras: first usable model (for the App Engineer)

GRU, 2 layers x 128 units, dropout 0.3, learning rate 1e-3 (run `gru_2x128_d0.3_lr0.001_s1`,
chosen on validation accuracy; see `results.csv` for all runs).

| | accuracy |
|---|---|
| validation (3 unseen participants) | 64.3% |
| test (3 unseen participants) | 83.8% |

Averaged over 3 seeds with the same settings: validation 63.5%, test 82.7%.
Expect it to be wrong fairly often on a new person, especially for the word `go`
(about 27-44% recall), and when the hands are only visible for a few frames.

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
  A mirrored input swaps left and right hands and the model will fail.
- Do not normalize or resample yourself: `preprocess_sample()` does it all.
- The training samples had a median of 22 original frames (range 2-308), resampled to 30.
  The buffer length is a choice to tune on live tests; `preprocess_sample` accepts any length.
- Requires TensorFlow 2.15.1 (see `requirements.txt`).
