from pathlib import Path
from evaluation.data_loader import load_test_data

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARRAYS_DIR = r"C:\Users\Dell\Documents\Study file\SignSinc\arrays-20261127T165626Z-1-001\arrays"

print("test_data_loader.py is running")

def test_load_test_data():
    X_test, y_test, groups_test, hand_frames_test = load_test_data(
        arrays_dir=ARRAYS_DIR
    )

    print("\n--- Test Dataset Information ---")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("groups_test shape:", groups_test.shape)
    print("hand_frames_test shape:", hand_frames_test.shape)

    print("Number of test samples:", len(X_test))
    print("Test participants:", sorted(set(groups_test.tolist())))

    assert X_test.shape == (814, 30, 144)
    assert y_test.shape == (814,)
    assert groups_test.shape == (814,)
    assert hand_frames_test.shape == (814,)

    print("\nTest data loaded successfully!")

if __name__ == "__main__":
    test_load_test_data()