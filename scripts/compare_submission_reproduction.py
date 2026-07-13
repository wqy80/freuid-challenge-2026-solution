from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    candidate = pd.read_csv(args.candidate)
    for name, frame in (("reference", reference), ("candidate", candidate)):
        if set(frame.columns) != {"id", "label"}:
            raise ValueError(f"{name} must contain exactly id,label")
        if frame["id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate ids")
    merged = reference.merge(candidate, on="id", suffixes=("_reference", "_candidate"), validate="one_to_one")
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("reference/candidate ID sets differ")
    delta = np.abs(
        merged["label_reference"].to_numpy(dtype=np.float64)
        - merged["label_candidate"].to_numpy(dtype=np.float64)
    )
    result = {
        "rows": len(merged),
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "rows_above_atol": int((delta > args.atol).sum()),
        "atol": args.atol,
        "status": "OK" if float(delta.max()) <= args.atol else "DIFF",
    }
    print(result)
    if result["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
