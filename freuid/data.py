from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_size(size: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(size, str):
        parts = size.lower().replace("x", ",").split(",")
        if len(parts) != 2:
            raise ValueError("--image-size must look like 640,1024 or 640x1024")
        return int(parts[0]), int(parts[1])
    return int(size[0]), int(size[1])


def build_transform(image_size: str | tuple[int, int], mode: Literal["train", "valid"]):
    height, width = parse_size(image_size)
    base = [
        transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ]
    if mode == "train":
        aug = [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15),
        ]
    else:
        aug = []
    return transforms.Compose(base + aug + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def resolve_train_path(data_root: Path, image_path: str) -> Path:
    rel = Path(image_path)
    candidates = [
        data_root / rel,
        data_root / rel.parts[0] / rel if rel.parts else data_root / rel,
        data_root / "train" / rel,
        data_root / "train" / "train" / rel.name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def public_test_path(test_dir: Path, image_id: str) -> Path:
    return test_dir / f"{image_id}.jpeg"


class FreuidTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str | Path, image_size: str | tuple[int, int], mode: Literal["train", "valid"]):
        self.df = df.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = build_transform(image_size, mode)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = resolve_train_path(self.data_root, row["image_path"])
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return {"image": x, "label": y, "id": row["id"]}


class FreuidTestDataset(Dataset):
    def __init__(self, ids: list[str], test_dir: str | Path, image_size: str | tuple[int, int]):
        self.ids = list(ids)
        self.test_dir = Path(test_dir)
        self.transform = build_transform(image_size, "valid")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        path = public_test_path(self.test_dir, image_id)
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "id": image_id}
