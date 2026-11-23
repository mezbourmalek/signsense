# **SignSense — Prepared Dataset**

**Owner:** Data Engineering  
**Dataset:** Google Isolated Sign Language Recognition (Google ISLR)  
**Source:** Kaggle — `asl-signs`  
**Release date:** 2026-11-21  
**Status:** Current dataset release

---

## **1\. Overview**

This document describes the prepared dataset used by the SignSense project.

The dataset contains landmark sequences representing **15 isolated American Sign Language (ASL) words**. Each sample corresponds to one signing sequence.

The prepared data is shared so that the Machine Learning Engineers and App Engineer can use the same data structure and preprocessing conventions.

**Dataset summary:**

| Property | Value |
| :---- | :---- |
| Selected words | 15 |
| Participants | 21 |
| Samples | 5,671 |
| Frames per sample | 30 |
| Features per frame | 144 |
| Data type of `X` | float32 |

> The dataset contains ASL signs. It does not represent Algerian Sign Language.

---

## **2\. Dataset Structure**

The prepared dataset is organized into two main folders:

```
SignSense/
│
├── README.md
│
├── arrays/
│   ├── X.npy
│   ├── y.npy
│   ├── groups.npy
│   ├── hand_frames.npy
│   ├── sequence_ids.npy
│   ├── labels.json
│   └── metadata.json
│
└── extracted/
    ├── frames.npy
    ├── index.csv
    ├── excluded.csv
    └── metadata.json
```

The original Kaggle dataset is not included in this shared folder. It remains stored separately.

### **2.1 `arrays/`**

This folder contains the final fixed-length arrays used by the Machine Learning Engineers.

| File | Shape / Content | Description |
| :---- | :---- | :---- |
| `X.npy` | `(5671, 30, 144)` | Input features for the models |
| `y.npy` | `(5671,)` | Integer label for each sample |
| `groups.npy` | `(5671,)` | Participant ID associated with each sample |
| `hand_frames.npy` | `(5671,)` | Number of original frames containing at least one detected hand |
| `sequence_ids.npy` | `(5671,)` | Original Kaggle sequence ID for traceability |
| `labels.json` | Label mapping | Maps each word to its integer label |
| `metadata.json` | Dataset information | Records the array configuration and preprocessing details |

**Important:** All files use the same sample order. For example, `X[i]`, `y[i]`, `groups[i]`, `hand_frames[i]`, and `sequence_ids[i]` refer to the same sample.

### **2.2 Understanding the main arrays**

* **`X.npy`**: The actual input data given to the model. It contains 5,671 samples, each represented by 30 frames and 144 values per frame.  
* **`y.npy`**: The correct word associated with each sample. It is the target the model learns to predict.  
* **`groups.npy`**: Stores the participant ID, not a model feature. It is used to separate participants between training, validation, and test sets.  
* **`hand_frames.npy`**: Records how many frames in the original sequence contained at least one detected hand, before resampling. It can help analyze sample quality.  
* **`sequence_ids.npy`**: Keeps the original Kaggle sequence ID, allowing a prepared sample to be traced back to its source.  
* **`labels.json`**: Provides the correspondence between integer labels and sign names.  
* **`metadata.json`**: Describes the dataset configuration, feature layout, and preprocessing settings.

---

## **3\. Label Mapping**

The dataset contains 15 selected words.

| Label | Word |
| :---- | :---- |
| 0 | hello |
| 1 | thankyou |
| 2 | please |
| 3 | yes |
| 4 | no |
| 5 | wait |
| 6 | finish |
| 7 | where |
| 8 | go |
| 9 | home |
| 10 | drink |
| 11 | book |
| 12 | airplane |
| 13 | car |
| 14 | jump |

The integer in `y.npy` identifies the corresponding word using this mapping.

---

## **4\. Feature Structure**

Each frame contains **144 values**, organized into three blocks:

| Feature block | Landmark count | Values |
| :---- | :---- | :---- |
| Left hand | 21 | 63 |
| Right hand | 21 | 63 |
| Pose | 6 | 18 |
| **Total** | **48** | **144** |

Each landmark contains three coordinates: `x`, `y`, and `z`.

The feature order is:

```
Frame (144 values)
│
├── Left hand  : 0–62
├── Right hand : 63–125
└── Pose       : 126–143
```

The six pose landmarks correspond to the following MediaPipe indices:

```
[11, 12, 13, 14, 15, 16]
```

These represent the shoulders, elbows, and wrists.

**Note:** The left/right hand block order and the correspondence with the live application's landmarks must remain consistent. The team should confirm the final convention before integrating the webcam pipeline.

---

## **5\. Preprocessing**

The final array `X.npy` is generated through the data preparation pipeline.

The preprocessing operations are:

1. **Normalization:** Hand and pose coordinates are normalized using the shoulder midpoint and shoulder width. The same scale convention is applied to the `z` coordinate.  
2. **Short-gap filling:** Missing hand values are filled only for gaps of up to 5 frames when valid frames exist on both sides.  
3. **Remaining missing values:** Any remaining `NaN` values are replaced with zero.  
4. **Resampling:** Each sequence is resampled to exactly 30 frames using nearest-neighbor resampling.

The resulting shape is:

```
X = (5671, 30, 144)
```

### **Important notes**

* The original sequence lengths are not necessarily 30 frames. Resampling makes all final sequences the same length.  
* `hand_frames.npy` preserves the original hand-frame count before resampling. It should not be interpreted as the number of visible-hand frames in the final array.  
* A zero-valued hand block in the processed array is an indicator of unavailable or zero-valued processed coordinates. It does not always prove that the hand was physically absent.

---

## **6\. Participant-Based Data Split**

The dataset is divided by **participant**, not by individual sample.

This means that all samples belonging to one participant must stay in only one subset.

The purpose is to evaluate whether the model can recognize signs from participants it did not see during training.

### **6.1 Split distribution**

| Subset | Participants | Samples |
| :---- | :---- | :---- |
| Training | 15 | 4,061 |
| Validation | 3 | 796 |
| Test | 3 | 814 |
| **Total** | **21** | **5,671** |

Each subset contains samples from all 15 words.

### **6.2 Participant IDs**

The split is defined in:

```
splits/split.json
```

The participant IDs are:

**Training:**

```
[2044, 4718, 16069, 18796, 27610,
 28656, 32319, 34503, 36257, 37055,
 37779, 49445, 53618, 55372, 62590]
```

**Validation:**

```
[22343, 29302, 30680]
```

**Test:**

```
[25571, 26734, 61333]
```

The split file must be used as defined. Do not create a new random split at sample level.

### **6.3 Why is this important?**

If samples from the same participant appear in both training and test sets, the model may learn participant-specific patterns.

This can lead to data leakage and an overly optimistic evaluation.

With participant-based splitting, the validation and test sets contain participants that are not present in the training set.

---

## **7\. Loading the Dataset**

The recommended way to load the data is through the shared `load_data.py` module in the GitHub repository.

Example:

```
from load_data import load_split

data = load_split(
    arrays_dir="path/to/arrays"
)

X_train, y_train = data["train"]
X_val, y_val = data["validation"]
X_test, y_test = data["test"]
```

The function:

* Loads the prepared arrays.  
* Reads the participant IDs from `splits/split.json`.  
* Selects samples according to their participant IDs.  
* Checks that participants do not overlap between subsets.  
* Checks that all samples are covered by the split.

By default, no minimum-hand-frame filtering is applied to the training set.

An optional parameter is available:

```
data = load_split(
    arrays_dir="path/to/arrays",
    min_hand_frames_train=5
)
```

This filters the training samples to keep only those with at least 5 original frames containing a detected hand.

**Important:** This option affects the training set only. Validation and test remain unchanged. The default value is `0`, so no such filtering occurs unless explicitly requested.

To retrieve participant IDs as well:

```
data = load_split(
    arrays_dir="path/to/arrays",
    with_groups=True
)

X_train, y_train, groups_train = data["train"]
```

The `groups` values are useful for participant-level analysis. They are not normally passed as model input features.

---

## **8\. Extracted Data**

The `extracted/` folder contains the landmark data before the final fixed-length preprocessing.

It is useful when the team needs to inspect original sequences or apply a different preprocessing or augmentation strategy.

| File | Description |
| :---- | :---- |
| `frames.npy` | Landmark values stored frame by frame |
| `index.csv` | Maps each sample to its position in `frames.npy` |
| `excluded.csv` | Records samples excluded during extraction |
| `metadata.json` | Describes the extracted data |

### **8.1 How `frames.npy` and `index.csv` work**

`frames.npy` stores the extracted frames in one continuous array.

`index.csv` tells us where each sample starts and how many frames it contains.

For a sample with:

* `start = 100`  
* `n_frames = 20`

its frames are:

```
sample = frames[100:120]
```

This means the extracted data keeps the original sequence length instead of forcing every sequence to 30 frames.

### **8.2 Excluded samples**

Two samples were excluded because their frame numbers were not consecutive:

| Participant ID | Sequence ID |
| :---- | :---- |
| 61333 | 940100783 |
| 30680 | 1125180648 |

The reasons and details are recorded in `excluded.csv`.

---

## **9\. Data Quality and Known Limitations**

The prepared dataset was checked for structural and numerical consistency.

The following points should be considered when training and evaluating models:

* The original sequences have different lengths. Their length ranges from 2 to 308 frames.  
* Approximately 62.6% of the original sequences contain fewer than 30 frames.  
* The median original sequence length is 22 frames.  
* Hand visibility varies considerably between samples and participants.  
* Some samples contain relatively few frames with detected hand landmarks.  
* The `car` class has 41 test samples, fewer than the other classes.

### **Hand-frame information**

`hand_frames.npy` provides a useful quality indicator.

Its median value is 14 original frames containing at least one detected hand.

Some words have a relatively high proportion of samples with fewer than 10 hand frames. This should be considered when interpreting model performance.

**Important limitation:** The hand-frame count is a quality indicator, not a direct measure of signing quality or model difficulty. A low count does not automatically mean that a sample is unusable.

The ML Engineers should inspect the relationship between sample quality and model errors before deciding whether to filter or weight samples.

---

## **10\. Responsibilities and Usage**

| Team role | Relevant files | Main purpose |
| :---- | :---- | :---- |
| ML Engineer 1 | `arrays/`, `load_data.py`, `features.py` | Load the data and train the main model |
| ML Engineer 2 | `arrays/`, `hand_frames.npy`, `labels.json` | Build baseline experiments, compare models, and analyze errors |
| App Engineer | `features.py` and the feature-layout documentation | Ensure that live webcam landmarks follow the same feature structure |
| Project Manager | `README.md`, `metadata.json`, split information | Track the data deliverable and its limitations |

### **Shared preprocessing**

The same feature conventions must be used during model training and live webcam inference.

The App Engineer should not independently change the feature order or normalization rules.

Any change to the shared feature structure should be discussed with the team first.

---

## **11\. Reproducing the Data Pipeline**

The data preparation scripts are maintained in the SignSense GitHub repository.

The raw Kaggle dataset must be available locally before rebuilding the prepared data.

The general pipeline is:

```
Original Kaggle Dataset
          |
          v
   Landmark Extraction
          |
          v
     extracted/
          |
          v
  Feature Preprocessing
          |
          v
       arrays/
          |
          v
 Participant-Based Split
          |
          v
   ML Training and Evaluation
```

The main scripts involved are:

| Script | Purpose |
| :---- | :---- |
| `config.py` | Stores shared configuration and data paths |
| `extract_landmarks.py` | Extracts landmark values from Parquet files |
| `features.py` | Contains shared feature preprocessing functions |
| `build_arrays.py` | Builds the final arrays |
| `load_data.py` | Loads arrays according to the participant split |

The exact commands and paths depend on the local configuration. The raw and processed datasets are stored outside the Git repository.

---

## **12\. Data Integrity**

A `checksums.txt` file can be added to the shared data folder to store SHA-256 hashes of the prepared files.

These hashes help verify that files have not changed or become corrupted during transfer.

This file is optional and is not used during model training.

---

## **13\. Important Rules**

1. Do not modify the participant split without team agreement.  
2. Do not perform a random sample-level split.  
3. Do not pass `groups`, `hand_frames`, or `sequence_ids` as ordinary model input features.  
4. Keep the feature order consistent across training and webcam inference.  
5. Do not commit the raw or processed dataset files to GitHub.  
6. Discuss changes to preprocessing or the feature layout with the team before applying them.  
7. Report model results together with the relevant data limitations.

---

**End of document — SignSense Data Engineering**

&nbsp;