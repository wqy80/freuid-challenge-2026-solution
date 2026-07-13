from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXPECTED = (
    "convnext_large_fold0.pt",
    "convnext_large_fold1.pt",
    "convnext_large_fold2.pt",
    "convnext_large_fold4.pt",
    "swinv2_large_fold0.pt",
    "swinv2_large_fold1.pt",
    "swinv2_large_fold2.pt",
    "synthetic_localization_fold2_epoch3.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--write-manifest", default="")
    args = parser.parse_args()
    root = Path(args.weights_dir)
    missing = [name for name in EXPECTED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen weights: {missing}")
    rows = []
    for name in EXPECTED:
        path = root / name
        rows.append({"file": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    if args.write_manifest:
        out = Path(args.write_manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "size_bytes", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
    print({"weights": len(rows), "total_bytes": sum(row["size_bytes"] for row in rows), "status": "OK"})


if __name__ == "__main__":
    main()
