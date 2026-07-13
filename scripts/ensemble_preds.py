import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freuid.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", action="append", required=True, help="Prediction CSV with id,label. Repeat for each model.")
    parser.add_argument("--weight", action="append", type=float, default=None, help="Optional model weight. Repeat once per --pred.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.weight is not None and len(args.weight) != len(args.pred):
        raise ValueError("--weight count must match --pred count")

    frames = []
    for path in args.pred:
        df = pd.read_csv(path)
        missing = {"id", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frames.append(df[["id", "label"]].copy())

    base_ids = frames[0]["id"].tolist()
    labels = []
    for path, df in zip(args.pred, frames):
        if df["id"].tolist() != base_ids:
            raise ValueError(f"{path} id order does not match first prediction")
        labels.append(df["label"].to_numpy(dtype=np.float64))

    pred_stack = np.vstack(labels)
    if args.weight is None:
        weights = np.ones(len(labels), dtype=np.float64) / len(labels)
    else:
        weights = np.asarray(args.weight, dtype=np.float64)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("--weight values must be non-negative and have positive sum")
        weights = weights / weights.sum()

    out = pd.DataFrame({"id": base_ids, "label": np.average(pred_stack, axis=0, weights=weights)})
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)} models={len(labels)} weights={weights.tolist()}")


if __name__ == "__main__":
    main()
