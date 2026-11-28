
"""
SignSense - Test Data Loader

Loads the fixed participant-based test split and its
corresponding metadata.

Dependencies:
    numpy
    load_data.py (from the project root)
"""

from pathlib import Path

import numpy as np

from load_data import DEFAULT_SPLIT, load_split


def load_test_data(arrays_dir, split_path=DEFAULT_SPLIT):
    """
    Load the fixed test set and its metadata.

    Returns:
        X_test         : (N, 30, 144)
        y_test         : (N,)
        groups_test    : (N,)
        hand_frames_test: (N,)
    """

    arrays_dir = Path(arrays_dir)
    split_path = Path(split_path)

    # 1. Load the fixed participant-based split
    data = load_split(
        arrays_dir=arrays_dir,
        split_path=split_path,
        with_groups=True
    )

    X_test, y_test, groups_test = data["test"]

    # 2. Load the complete metadata arrays
    all_groups = np.load(arrays_dir / "groups.npy")
    all_hand_frames = np.load(arrays_dir / "hand_frames.npy")
    all_X = np.load(arrays_dir / "X.npy")

    # 3. Check that metadata arrays have matching lengths
    if not (
        len(all_groups) == len(all_hand_frames) == len(all_X)
    ):
        raise ValueError(
            "Inconsistent sample counts between dataset arrays."
        )

    # 4. Select hand_frames belonging to test participants
    # np.isin preserves the original sample order.
    test_mask = np.isin(all_groups, np.unique(groups_test))

    hand_frames_test = all_hand_frames[test_mask]

    # 5. Verify that the selected samples match load_split()
    if not np.array_equal(all_X[test_mask], X_test):
        raise ValueError(
            "Test samples do not match the selected participant split."
        )

    if not np.array_equal(all_groups[test_mask], groups_test):
        raise ValueError(
            "Participant IDs are not aligned with the test samples."
        )

    # 6. Verify the expected input dimensions
    if X_test.ndim != 3 or X_test.shape[1:] != (30, 144):
        raise ValueError(
            f"Unexpected X_test shape: {X_test.shape}. "
            "Expected (N, 30, 144)."
        )

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different lengths.")

    if not (
        len(X_test) == len(groups_test) == len(hand_frames_test)
    ):
        raise ValueError("Test data and metadata are not aligned.")

    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains NaN or infinite values.")

    # 7. Display a short loading summary
    print("Test data loaded successfully.")
    print(f"X_test shape       : {X_test.shape}")
    print(f"y_test shape       : {y_test.shape}")
    print(f"Participants       : {np.unique(groups_test).tolist()}")
    print(f"Test samples       : {len(X_test)}")
    print(f"hand_frames shape  : {hand_frames_test.shape}")

    return X_test, y_test, groups_test, hand_frames_test
