from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    image_ids = sorted(path.stem for path in Path(args.image_dir).glob("*.jpeg"))
    frame = pd.read_csv(args.submission)
    if list(frame.columns) != ["id", "label"]:
        raise ValueError(f"expected id,label columns; got {frame.columns.tolist()}")
    if frame["id"].duplicated().any():
        raise ValueError("duplicate submission ids")
    if frame["id"].astype(str).tolist() != image_ids:
        raise ValueError("submission IDs do not exactly match sorted image filenames")
    values = frame["label"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("submission labels must be finite values in [0, 1]")
    print({"rows": len(frame), "min": float(values.min()), "max": float(values.max()), "status": "OK"})


if __name__ == "__main__":
    main()
