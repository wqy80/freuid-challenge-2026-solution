from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def add_folds(df: pd.DataFrame, n_splits: int, seed: int, mode: str) -> pd.DataFrame:
    df = df.copy()
    y = df["label"].astype(int).values

    if mode == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(df)), y)
    elif mode == "type_group":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(df)), y, groups=df["type"].values)
    else:
        raise ValueError(f"unknown split mode: {mode}")

    fold_col = f"fold_{mode}"
    df[fold_col] = -1
    for fold, (_, valid_idx) in enumerate(iterator):
        df.loc[df.index[valid_idx], fold_col] = fold
    return df
