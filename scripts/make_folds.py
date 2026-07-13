import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freuid.paths import DEFAULT_DATA_ROOT, DEFAULT_TRAIN_CSV
from freuid.splits import add_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--out", default=str(DEFAULT_DATA_ROOT / "train_labels_folds.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    df = add_folds(df, args.n_splits, args.seed, "stratified")
    df = add_folds(df, args.n_splits, args.seed, "type_group")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} rows={len(df)} columns={list(df.columns)}")
    print(df[["label", "type", "fold_stratified", "fold_type_group"]].groupby(["fold_stratified", "label"]).size())


if __name__ == "__main__":
    main()
