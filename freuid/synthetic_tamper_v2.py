from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from freuid.data import IMAGENET_MEAN, IMAGENET_STD, resolve_train_path


# Normalized field regions follow the five document templates in FREUID train/public.
FIELD_REGIONS = {
    "MAURITIUS/ID": {
        "portrait": (0.035, 0.20, 0.35, 0.82),
        "text": (0.36, 0.19, 0.70, 0.58),
        "signature": (0.34, 0.68, 0.68, 0.86),
    },
    "BENIN/DL": {
        "portrait": (0.025, 0.22, 0.34, 0.80),
        "text": (0.35, 0.22, 0.86, 0.71),
        "signature": (0.43, 0.73, 0.78, 0.91),
    },
    "MOZAMBIQUE/DL": {
        "portrait": (0.025, 0.22, 0.30, 0.72),
        "text": (0.30, 0.15, 0.88, 0.70),
        "signature": (0.28, 0.58, 0.57, 0.72),
    },
    "GUINEA/DL": {
        "portrait": (0.035, 0.20, 0.34, 0.78),
        "text": (0.33, 0.18, 0.88, 0.72),
        "signature": (0.30, 0.72, 0.62, 0.91),
    },
    "EGYPT/DL": {
        "portrait": (0.03, 0.20, 0.33, 0.78),
        "text": (0.32, 0.17, 0.88, 0.72),
        "signature": (0.30, 0.70, 0.66, 0.90),
    },
}


@dataclass(frozen=True)
class SyntheticTamperV2Config:
    image_size: tuple[int, int] = (896, 1408)
    synth_probability: float = 0.30
    seed: int = 42


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=int(quality), subsampling=2)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB")


def _pixel_box(region, width: int, height: int, rng: random.Random, kind: str):
    rx0, ry0, rx1, ry1 = region
    region_w = max(4, int((rx1 - rx0) * width))
    region_h = max(4, int((ry1 - ry0) * height))
    if kind == "text":
        bw = rng.randint(max(18, int(region_w * 0.38)), max(19, int(region_w * 0.92)))
        bh = rng.randint(max(8, int(region_h * 0.08)), max(9, int(region_h * 0.20)))
    elif kind == "signature":
        bw = rng.randint(max(18, int(region_w * 0.55)), max(19, int(region_w * 0.95)))
        bh = rng.randint(max(8, int(region_h * 0.28)), max(9, int(region_h * 0.75)))
    else:
        bw = rng.randint(max(24, int(region_w * 0.72)), max(25, int(region_w * 0.98)))
        bh = rng.randint(max(24, int(region_h * 0.72)), max(25, int(region_h * 0.98)))
    x_min, y_min = int(rx0 * width), int(ry0 * height)
    x_max, y_max = max(x_min, int(rx1 * width) - bw), max(y_min, int(ry1 * height) - bh)
    x0 = rng.randint(x_min, x_max)
    y0 = rng.randint(y_min, y_max)
    return x0, y0, x0 + bw, y0 + bh


def _match_color(patch: Image.Image, target: Image.Image) -> Image.Image:
    source = np.asarray(patch.convert("RGB"), dtype=np.float32)
    reference = np.asarray(target.convert("RGB"), dtype=np.float32)
    source_mean = source.mean(axis=(0, 1), keepdims=True)
    reference_mean = reference.mean(axis=(0, 1), keepdims=True)
    source_std = source.std(axis=(0, 1), keepdims=True).clip(4.0, None)
    reference_std = reference.std(axis=(0, 1), keepdims=True).clip(4.0, 60.0)
    matched = (source - source_mean) * (reference_std / source_std) + reference_mean
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))


def synthesize_field_tamper(image: Image.Image, donor: Image.Image, doc_type: str, rng: random.Random):
    image = image.convert("RGB")
    donor = donor.convert("RGB").resize(image.size, Image.Resampling.BICUBIC)
    width, height = image.size
    regions = FIELD_REGIONS.get(doc_type, FIELD_REGIONS["BENIN/DL"])
    choice = rng.random()
    kind = "text" if choice < 0.50 else "portrait" if choice < 0.82 else "signature"
    box = _pixel_box(regions[kind], width, height, rng, kind)
    target = image.crop(box)
    patch = donor.crop(box).resize(target.size, Image.Resampling.BICUBIC)
    patch = _match_color(patch, target)
    patch = ImageEnhance.Contrast(patch).enhance(rng.uniform(0.96, 1.04))
    patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(0.97, 1.03))

    # Weak edge preserves realistic compositing; whole-image JPEG removes trivial seams.
    alpha_level = rng.randint(225, 255)
    alpha = Image.new("L", target.size, 0)
    feather = rng.randint(1, max(1, min(target.size) // 25))
    ImageDraw.Draw(alpha).rectangle(
        (feather, feather, max(feather, target.width - feather - 1), max(feather, target.height - feather - 1)),
        fill=alpha_level,
    )
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.35, 1.2)))
    output = image.copy()
    output.paste(patch, box[:2], alpha)
    output = _jpeg(output, rng.randint(72, 96))
    mask = Image.new("L", image.size, 0)
    mask.paste(255, box)
    return output, mask, f"same_template_{kind}"


class SyntheticTamperV2Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str | Path, config: SyntheticTamperV2Config, mode: str, max_samples: int = 0):
        if mode not in {"train", "valid", "synthetic_valid"}:
            raise ValueError(mode)
        frame = df.reset_index(drop=True).copy()
        if mode == "synthetic_valid":
            frame = frame[frame["label"].astype(int) == 0].reset_index(drop=True)
            if max_samples > 0 and len(frame) > max_samples:
                frame = frame.sample(max_samples, random_state=config.seed).reset_index(drop=True)
        self.df = frame
        self.data_root = Path(data_root)
        self.config = config
        self.mode = mode
        self.height, self.width = config.image_size
        genuine = df[df["label"].astype(int) == 0].reset_index(drop=True)
        self.group_donors = {}
        for _, row in genuine.iterrows():
            key = (str(row.get("type", "")), str(row.get("is_digital", "")))
            self.group_donors.setdefault(key, []).append(resolve_train_path(self.data_root, str(row["image_path"])))
        self.type_donors = {}
        for key, paths in self.group_donors.items():
            self.type_donors.setdefault(key[0], []).extend(paths)
        self.all_donors = [path for paths in self.group_donors.values() for path in paths]

    def __len__(self):
        return len(self.df)

    def _rng(self, idx):
        return random.Random(self.config.seed * 1_000_003 + idx) if self.mode == "synthetic_valid" else random

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_path = resolve_train_path(self.data_root, str(row["image_path"]))
        image = Image.open(source_path).convert("RGB")
        rng = self._rng(idx)
        original_label = int(row["label"])
        synthesize = self.mode == "synthetic_valid" or (
            self.mode == "train" and original_label == 0 and rng.random() < self.config.synth_probability
        )
        operation = "native"
        mask = Image.new("L", image.size, 0)
        label = original_label
        if synthesize:
            doc_type = str(row.get("type", ""))
            key = (doc_type, str(row.get("is_digital", "")))
            donors = self.group_donors.get(key) or self.type_donors.get(doc_type) or self.all_donors
            donor_path = donors[rng.randrange(len(donors))]
            for _ in range(10):
                if donor_path != source_path or len(donors) == 1:
                    break
                donor_path = donors[rng.randrange(len(donors))]
            donor = Image.open(donor_path).convert("RGB")
            image, mask, operation = synthesize_field_tamper(image, donor, doc_type, rng)
            label = 1

        image = image.resize((self.width, self.height), Image.Resampling.BICUBIC)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)
        if self.mode == "train" and rng.random() < 0.20:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.95, 1.05))
        tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return {
            "image": tensor,
            "mask": TF.to_tensor(mask).clamp(0, 1),
            "mask_valid": torch.tensor(float(synthesize), dtype=torch.float32),
            "label": torch.tensor(float(label), dtype=torch.float32),
            "original_label": torch.tensor(float(original_label), dtype=torch.float32),
            "id": str(row["id"]),
            "operation": operation,
        }
