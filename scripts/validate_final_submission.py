from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--predicted-ids", default="")
    args = parser.parse_args()

    submission = pd.read_csv(args.submission)
    template = pd.read_csv(args.template)
    if list(submission.columns) != ["id", "label"]:
        raise ValueError(f"submission columns must be exactly id,label; got {submission.columns.tolist()}")
    if "id" not in template.columns:
        raise ValueError("template has no id column")
    if submission["id"].duplicated().any() or template["id"].duplicated().any():
        raise ValueError("duplicate ids detected")
    submission["id"] = submission["id"].astype(str)
    template["id"] = template["id"].astype(str)
    if submission["id"].tolist() != template["id"].tolist():
        raise ValueError("submission id order/content does not exactly match template")
    values = submission["label"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("submission contains NaN or infinite labels")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("submission labels are outside [0, 1]")

    predicted = 0
    if args.predicted_ids:
        pred = pd.read_csv(args.predicted_ids)
        if "id" not in pred.columns or pred["id"].duplicated().any():
            raise ValueError("predicted-ids CSV must contain unique id values")
        unknown = set(pred["id"].astype(str)) - set(template["id"])
        if unknown:
            raise ValueError(f"predicted IDs contain {len(unknown)} unknown values")
        predicted = len(pred)

    print(
        {
            "submission": str(Path(args.submission)),
            "rows": len(submission),
            "predicted_rows_this_run": predicted,
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "exact_zero": int((values == 0).sum()),
            "exact_one": int((values == 1).sum()),
            "status": "OK",
        }
    )


if __name__ == "__main__":
    main()
