from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from freuid.data import IMAGENET_MEAN, IMAGENET_STD, resolve_train_path
from freuid.synthetic_tamper import TAMPER_OPERATIONS, synthesize_tamper


@dataclass(frozen=True)
class SyntheticTamperV3Config:
    image_size: tuple[int, int] = (896, 1408)
    synth_probability: float = 0.35
    recapture_probability: float = 0.50
    seed: int = 42


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _perspective_pair(image: Image.Image, mask: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    magnitude = rng.uniform(0.008, 0.035)
    dx, dy = width * magnitude, height * magnitude
    start = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    end = [
        [rng.uniform(0, dx), rng.uniform(0, dy)],
        [width - 1 - rng.uniform(0, dx), rng.uniform(0, dy)],
        [width - 1 - rng.uniform(0, dx), height - 1 - rng.uniform(0, dy)],
        [rng.uniform(0, dx), height - 1 - rng.uniform(0, dy)],
    ]
    image = TF.perspective(
        image,
        start,
        end,
        interpolation=InterpolationMode.BICUBIC,
        fill=[245, 245, 245],
    )
    mask = TF.perspective(mask, start, end, interpolation=InterpolationMode.NEAREST, fill=0)
    return image, mask


def _sensor_and_moire(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    sigma = rng.uniform(0.8, 3.5)
    array += np.random.default_rng(rng.randrange(2**32)).normal(0.0, sigma, array.shape).astype(np.float32)

    if rng.random() < 0.65:
        yy, xx = np.mgrid[0:height, 0:width]
        angle = rng.uniform(0.0, math.pi)
        period = rng.uniform(5.0, 18.0)
        amplitude = rng.uniform(0.6, 3.5)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        pattern = amplitude * np.sin((math.cos(angle) * xx + math.sin(angle) * yy) * (2.0 * math.pi / period) + phase)
        array += pattern[..., None]

    if rng.random() < 0.55:
        yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        center_x, center_y = rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)
        radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        shade = 1.0 - rng.uniform(0.02, 0.10) * np.clip(radius, 0.0, 1.5)
        array *= shade[..., None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def simulate_recapture(image: Image.Image, mask: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    image, mask = _perspective_pair(image.convert("RGB"), mask.convert("L"), rng)
    width, height = image.size

    scale = rng.uniform(0.45, 0.82)
    down_size = (max(96, int(width * scale)), max(64, int(height * scale)))
    image = image.resize(down_size, Image.Resampling.LANCZOS).resize((width, height), Image.Resampling.BICUBIC)
    mask = mask.resize(down_size, Image.Resampling.NEAREST).resize((width, height), Image.Resampling.NEAREST)

    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.78, 1.20))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.16))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.72, 1.20))
    if rng.random() < 0.75:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 1.35)))
    image = _sensor_and_moire(image, rng)
    image = _jpeg_roundtrip(image, rng.randint(28, 82))
    return image, mask


class SyntheticTamperV3Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str | Path, config: SyntheticTamperV3Config, mode: str):
        if mode not in {"train", "valid", "synthetic_valid"}:
            raise ValueError(f"unsupported mode: {mode}")
        frame = df.reset_index(drop=True).copy()
        if mode == "synthetic_valid":
            frame = frame[frame["label"].astype(int) == 0].reset_index(drop=True)
        self.df = frame
        self.data_root = Path(data_root)
        self.config = config
        self.mode = mode
        self.height, self.width = config.image_size
        genuine = df[df["label"].astype(int) == 0].reset_index(drop=True)
        self.donor_paths = [resolve_train_path(self.data_root, value) for value in genuine["image_path"].astype(str)]

    def __len__(self):
        return len(self.df)

    def _rng(self, index: int) -> random.Random:
        if self.mode == "synthetic_valid":
            return random.Random(self.config.seed * 1_000_003 + index)
        return random

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        with Image.open(resolve_train_path(self.data_root, str(row["image_path"]))) as source:
            image = source.convert("RGB")
        rng = self._rng(index)
        original_label = int(row["label"])
        synthetic = self.mode == "synthetic_valid" or (
            self.mode == "train" and original_label == 0 and rng.random() < self.config.synth_probability
        )
        mask = Image.new("L", image.size, 0)
        label = original_label
        operation = "native"
        if synthetic:
            with Image.open(self.donor_paths[rng.randrange(len(self.donor_paths))]) as donor_source:
                donor = donor_source.convert("RGB")
            forced = TAMPER_OPERATIONS[index % len(TAMPER_OPERATIONS)] if self.mode == "synthetic_valid" else None
            image, mask, operation = synthesize_tamper(image, donor, rng, forced)
            label = 1

        recaptured = self.mode == "synthetic_valid" or (
            self.mode == "train" and rng.random() < self.config.recapture_probability
        )
        if recaptured:
            image, mask = simulate_recapture(image, mask, rng)
            operation = f"{operation}+recapture"

        image = image.resize((self.width, self.height), Image.Resampling.BICUBIC)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = TF.to_tensor(mask).clamp(0, 1)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "mask_valid": torch.tensor(float(synthetic), dtype=torch.float32),
            "label": torch.tensor(float(label), dtype=torch.float32),
            "original_label": torch.tensor(float(original_label), dtype=torch.float32),
            "id": str(row["id"]),
            "operation": operation,
        }
