from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def read_prediction(path: str, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"id", "label"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain id,label")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate ids")
    frame = frame[["id", "label"]].copy()
    frame["id"] = frame["id"].astype(str)
    frame = frame.rename(columns={"label": name})
    values = frame[name].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{path} contains invalid scores")
    return frame


def normalized_blend(primary: np.ndarray, auxiliary: np.ndarray, auxiliary_weight) -> np.ndarray:
    weight = np.asarray(auxiliary_weight, dtype=np.float64)
    if np.any(weight < 0):
        raise ValueError("fusion weights must be non-negative")
    return (primary + weight * auxiliary) / (1.0 + weight)


def image_resolution(test_dir: Path, image_id: str) -> str:
    path = test_dir / f"{image_id}.jpeg"
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return f"{image.width}x{image.height}"


def resolution_weight(resolution: str, default: float, exact: dict[str, float], buckets) -> float:
    if resolution in exact:
        return exact[resolution]
    width, height = (int(value) for value in resolution.split("x", 1))
    short_side = min(width, height)
    for rule in sorted(buckets, key=lambda item: int(item["max_short_side"])):
        if short_side <= int(rule["max_short_side"]):
            return float(rule["weight"])
    return default


def write_prediction(ids: pd.Series, values: np.ndarray, path: Path) -> None:
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"candidate {path.name} contains invalid scores")
    pd.DataFrame({"id": ids, "label": values}).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convnext-pred", required=True)
    parser.add_argument("--swinv2-pred", required=True)
    parser.add_argument("--localization-pred", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selected-out", default="")
    parser.add_argument("--select", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    convnext = read_prediction(args.convnext_pred, "convnext")
    swinv2 = read_prediction(args.swinv2_pred, "swinv2")
    localization = read_prediction(args.localization_pred, "localization")
    merged = convnext.merge(swinv2, on="id", validate="one_to_one")
    merged = merged.merge(localization, on="id", validate="one_to_one")
    if len(merged) != len(convnext) or len(merged) != len(swinv2) or len(merged) != len(localization):
        raise ValueError("prediction ID sets differ")

    test_dir = Path(args.test_dir)
    merged["resolution"] = [image_resolution(test_dir, image_id) for image_id in merged["id"]]

    full_weights = config["fullimage_weights"]
    convnext_weight = float(full_weights["convnext"])
    swinv2_weight = float(full_weights["swinv2"])
    if convnext_weight <= 0 or swinv2_weight < 0:
        raise ValueError("invalid full-image weights")
    private_base = (
        convnext_weight * merged["convnext"].to_numpy(dtype=np.float64)
        + swinv2_weight * merged["swinv2"].to_numpy(dtype=np.float64)
    ) / (convnext_weight + swinv2_weight)

    resolution_config = config["resolution_safe"]
    default_swinv2_weight = float(resolution_config["default_swinv2_weight"])
    overrides = {
        str(key): float(value)
        for key, value in resolution_config.get("swinv2_weight_by_resolution", {}).items()
    }
    buckets = resolution_config.get("swinv2_weight_by_short_side", [])
    resolution_weights = np.asarray(
        [
            resolution_weight(resolution, default_swinv2_weight, overrides, buckets)
            for resolution in merged["resolution"]
        ],
        dtype=np.float64,
    )
    private_resolution_safe = normalized_blend(
        merged["convnext"].to_numpy(dtype=np.float64),
        merged["swinv2"].to_numpy(dtype=np.float64),
        resolution_weights,
    )

    localization_weight = float(config["localization_weight"])
    localization_values = merged["localization"].to_numpy(dtype=np.float64)
    private_localization = normalized_blend(private_base, localization_values, localization_weight)
    private_resolution_localization = normalized_blend(
        private_resolution_safe, localization_values, localization_weight
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "private_base": private_base,
        "private_resolution_safe": private_resolution_safe,
        "private_localization": private_localization,
        "private_resolution_localization": private_resolution_localization,
    }
    for name, values in candidates.items():
        write_prediction(merged["id"], values, out_dir / f"{name}.csv")

    diagnostics = merged.copy()
    diagnostics["swinv2_weight"] = resolution_weights
    for name, values in candidates.items():
        diagnostics[name] = values
    diagnostics.to_csv(out_dir / "candidate_diagnostics.csv", index=False)

    selected = args.select or str(config["default_candidate"])
    if selected not in candidates:
        raise ValueError(f"unknown selected candidate: {selected}")
    if args.selected_out:
        selected_out = Path(args.selected_out)
        selected_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_dir / f"{selected}.csv", selected_out)

    print(
        {
            "rows": len(merged),
            "candidates": list(candidates),
            "selected": selected,
            "resolution_counts": merged["resolution"].value_counts().to_dict(),
            "out_dir": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()
