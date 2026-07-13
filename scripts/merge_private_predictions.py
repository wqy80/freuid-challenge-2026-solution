from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_unique(path: str, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate ids")
    frame["id"] = frame["id"].astype(str)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--new-pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    template = read_unique(args.template, {"id"})[["id"]]
    base = read_unique(args.base_submission, {"id", "label"})[["id", "label"]]
    pred = read_unique(args.new_pred, {"id", "label"})[["id", "label"]]
    template_ids = set(template["id"])
    base_ids = set(base["id"])
    pred_ids = set(pred["id"])
    if base_ids != template_ids:
        raise ValueError(
            f"base submission/template mismatch: missing={len(template_ids-base_ids)} extra={len(base_ids-template_ids)}"
        )
    unknown = pred_ids - template_ids
    if unknown:
        raise ValueError(f"new predictions contain {len(unknown)} ids absent from template")
    if not np.isfinite(pred["label"].to_numpy(dtype=np.float64)).all():
        raise ValueError("new predictions contain non-finite values")

    merged = template.merge(base.rename(columns={"label": "base_label"}), on="id", validate="one_to_one")
    merged = merged.merge(pred.rename(columns={"label": "new_label"}), on="id", how="left", validate="one_to_one")
    replaced = int(merged["new_label"].notna().sum())
    merged["label"] = merged["new_label"].fillna(merged["base_label"]).clip(0.0, 1.0)
    output = merged[["id", "label"]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(
        f"wrote {out_path} rows={len(output)} replaced={replaced} "
        f"preserved={len(output)-replaced} overlap_with_base={len(pred_ids & base_ids)}"
    )


if __name__ == "__main__":
    main()
