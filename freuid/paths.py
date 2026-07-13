import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path(os.environ.get("FREUID_DATA_ROOT", "data"))
DEFAULT_TRAIN_CSV = DEFAULT_DATA_ROOT / "train_labels.csv"
DEFAULT_SUBMISSION_CSV = DEFAULT_DATA_ROOT / "sample_submission.csv"
DEFAULT_TRAIN_DIR = DEFAULT_DATA_ROOT / "train" / "train"
DEFAULT_PUBLIC_TEST_DIR = DEFAULT_DATA_ROOT / "public_test" / "public_test"
