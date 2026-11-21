"""
SignSense - Central Configuration Module

Purpose:
    Defines the shared constants and configuration settings
    used throughout the SignSense project.

Configuration sections:
    1. Vocabulary: The 15 target ASL words.
    2. Data specifications: Sequence length and landmark dimensions.
    3. Feature layout: Organization of hand and pose landmarks.

Design principles:
    - Maintain a single source of truth for shared constants.
    - Avoid duplicating configuration values across modules.
    - Ensure consistency across different project components.
    - Make configuration updates simple and centralized.

Example of usage:
    Import the required constants into another module:

        from config import WORDS, SEQUENCE_LENGTH, FEATURES_PER_FRAME

        print(WORDS)
        print(SEQUENCE_LENGTH)
        print(FEATURES_PER_FRAME)

    This allows different modules to use the same configuration
    without redefining constants.
"""
import os
from pathlib import Path

# --- Vocabulary (contract) ---------------------------------------------------
WORDS = ["hello", "thankyou", "please", "yes", "no", "wait", "finish", "where",
         "go", "home", "drink", "book", "airplane", "car", "jump"]

# --- Array shapes (contract) -------------------------------------------------
SEQUENCE_LENGTH = 30
HAND_LANDMARKS = 21
POSE_INDICES = [11, 12, 13, 14, 15, 16]  #  shoulders, elbows, wrists (L, R)
N_COORDS = 3  # x, y, z
FEATURES_PER_FRAME = (2 * HAND_LANDMARKS + len(POSE_INDICES)) * N_COORDS  # 144

# --- Layout inside the 144 values
LEFT_HAND_SLICE = slice(0, 63)
RIGHT_HAND_SLICE = slice(63, 126)
POSE_SLICE = slice(126, 144)

# --- Paths (outside the repository) ------------------------------------------
DATA_DIR = Path(os.environ.get("SIGNSENSE_DATA", r"E:\SignSense_data"))
TRAIN_CSV = DATA_DIR / "raw" / "train.csv"
LANDMARKS_DIR = DATA_DIR / "raw" / "landmarks"
PROCESSED_DIR = DATA_DIR / "processed"
MAX_GAP_FRAMES = 5  # longest run of missing hand frames that is filled by interpolation