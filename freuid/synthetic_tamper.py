from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from freuid.data import IMAGENET_MEAN, IMAGENET_STD, resolve_train_path


TAMPER_OPERATIONS = (
    "copy_move",
    "donor_splice",
    "text_replace",
    "photo_replace",
    "local_jpeg",
    "local_inpaint",
)


@dataclass(frozen=True)
class SyntheticTamperConfig:
    image_size: tuple[int, int] = (672, 1056)
    synth_probability: float = 0.35
    jpeg_probability: float = 0.35
    seed: int = 42


def _rand_box(rng: random.Random, width: int, height: int, operation: str) -> tuple[int, int, int, int]:
    if operation == "text_replace":
        bw = rng.randint(max(24, int(width * 0.10)), max(25, int(width * 0.38)))
        bh = rng.randint(max(10, int(height * 0.025)), max(11, int(height * 0.09)))
    elif operation == "photo_replace":
        bw = rng.randint(max(32, int(width * 0.13)), max(33, int(width * 0.30)))
        bh = rng.randint(max(32, int(height * 0.20)), max(33, int(height * 0.55)))
    else:
        bw = rng.randint(max(24, int(width * 0.07)), max(25, int(width * 0.28)))
        bh = rng.randint(max(18, int(height * 0.05)), max(19, int(height * 0.24)))
    bw = min(bw, width - 2)
    bh = min(bh, height - 2)
    x0 = rng.randint(1, max(1, width - bw - 1))
    y0 = rng.randint(1, max(1, height - bh - 1))
    return x0, y0, x0 + bw, y0 + bh


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _feather_mask(size: tuple[int, int], radius: float) -> Image.Image:
    mask = Image.new("L", size, 255)
    if radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))
    return mask


def synthesize_tamper(
    image: Image.Image,
    donor: Image.Image,
    rng: random.Random,
    operation: str | None = None,
) -> tuple[Image.Image, Image.Image, str]:
    image = image.convert("RGB")
    donor = donor.convert("RGB")
    width, height = image.size
    operation = operation or rng.choice(TAMPER_OPERATIONS)
    dst_box = _rand_box(rng, width, height, operation)
    x0, y0, x1, y1 = dst_box
    patch_size = (x1 - x0, y1 - y0)

    if operation == "copy_move":
        src_box = _rand_box(rng, width, height, operation)
        patch = image.crop(src_box).resize(patch_size, Image.Resampling.BICUBIC)
        if rng.random() < 0.5:
            patch = ImageOps.mirror(patch)
    elif operation in {"donor_splice", "text_replace", "photo_replace"}:
        dw, dh = donor.size
        src_operation = "text_replace" if operation == "text_replace" else "photo_replace" if operation == "photo_replace" else "copy_move"
        src_box = _rand_box(rng, dw, dh, src_operation)
        patch = donor.crop(src_box).resize(patch_size, Image.Resampling.BICUBIC)
        patch = ImageEnhance.Contrast(patch).enhance(rng.uniform(0.85, 1.15))
        patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(0.88, 1.12))
    elif operation == "local_jpeg":
        patch = image.crop(dst_box)
        patch = _jpeg_roundtrip(patch, rng.randint(18, 55))
    elif operation == "local_inpaint":
        patch = image.crop(dst_box)
        radius = rng.uniform(2.0, 7.0)
        patch = patch.filter(ImageFilter.GaussianBlur(radius=radius))
        patch = ImageEnhance.Contrast(patch).enhance(rng.uniform(0.75, 1.05))
    else:
        raise ValueError(f"unknown tamper operation: {operation}")

    alpha = _feather_mask(patch_size, rng.uniform(0.0, 2.2))
    alpha = alpha.point(lambda value: int(value * rng.uniform(0.78, 1.0)))
    output = image.copy()
    output.paste(patch, (x0, y0), alpha)
    if rng.random() < 0.35:
        output = _jpeg_roundtrip(output, rng.randint(55, 92))

    target = Image.new("L", image.size, 0)
    target.paste(255, dst_box)
    return output, target, operation


class SyntheticTamperDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str | Path,
        config: SyntheticTamperConfig,
        mode: str,
        max_samples: int = 0,
    ):
        if mode not in {"train", "valid", "synthetic_valid"}:
            raise ValueError(f"unsupported mode: {mode}")
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
        self.donor_paths = [resolve_train_path(self.data_root, value) for value in genuine["image_path"].astype(str)]

    def __len__(self) -> int:
        return len(self.df)

    def _rng(self, idx: int) -> random.Random:
        if self.mode == "synthetic_valid":
            return random.Random(self.config.seed * 1_000_003 + idx)
        return random

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(resolve_train_path(self.data_root, str(row["image_path"]))).convert("RGB")
        rng = self._rng(idx)
        original_label = int(row["label"])
        synthesize = self.mode == "synthetic_valid" or (
            self.mode == "train" and original_label == 0 and rng.random() < self.config.synth_probability
        )
        operation = "native"
        mask = Image.new("L", image.size, 0)
        label = original_label
        if synthesize:
            donor_path = self.donor_paths[rng.randrange(len(self.donor_paths))]
            donor = Image.open(donor_path).convert("RGB")
            forced_op = TAMPER_OPERATIONS[idx % len(TAMPER_OPERATIONS)] if self.mode == "synthetic_valid" else None
            image, mask, operation = synthesize_tamper(image, donor, rng, forced_op)
            label = 1

        image = image.resize((self.width, self.height), Image.Resampling.BICUBIC)
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)
        if self.mode == "train":
            if rng.random() < 0.30:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
            if rng.random() < 0.30:
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.08))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = TF.to_tensor(mask).clamp(0, 1)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "mask_valid": torch.tensor(float(synthesize), dtype=torch.float32),
            "label": torch.tensor(float(label), dtype=torch.float32),
            "original_label": torch.tensor(float(original_label), dtype=torch.float32),
            "id": str(row["id"]),
            "operation": operation,
        }


class ConvNeXtFpnLocalization(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_base.fb_in22k_ft_in1k",
        pretrained: bool = True,
        fpn_channels: int = 128,
        topk_fraction: float = 0.01,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required") from exc
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=3,
        )
        channels = self.encoder.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(channel, fpn_channels, 1) for channel in channels])
        self.smooth = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, fpn_channels),
            nn.GELU(),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(fpn_channels // 2, 1, 1),
        )
        self.score_head = nn.Linear(3, 1)
        with torch.no_grad():
            self.score_head.weight.copy_(torch.tensor([[0.55, 0.30, 0.15]]))
            self.score_head.bias.zero_()
        self.topk_fraction = float(topk_fraction)

    def _image_logit(self, heatmap_logits: torch.Tensor) -> torch.Tensor:
        flat = heatmap_logits.flatten(1)
        n = flat.shape[1]
        fractions = (self.topk_fraction / 2.0, self.topk_fraction, min(self.topk_fraction * 5.0, 1.0))
        stats = []
        for fraction in fractions:
            k = max(1, min(n, int(math.ceil(n * fraction))))
            stats.append(torch.topk(flat, k=k, dim=1).values.mean(dim=1))
        return self.score_head(torch.stack(stats, dim=1)).flatten()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(image)
        pyramid = [None] * len(features)
        pyramid[-1] = self.lateral[-1](features[-1])
        for index in range(len(features) - 2, -1, -1):
            up = F.interpolate(pyramid[index + 1], size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            pyramid[index] = self.lateral[index](features[index]) + up
        heatmap_logits = self.heatmap_head(self.smooth(pyramid[0]))
        image_logits = self._image_logit(heatmap_logits)
        return image_logits, heatmap_logits


def localization_loss(
    image_logits: torch.Tensor,
    heatmap_logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    mask_valid: torch.Tensor,
    image_weight: float,
    mask_bce_weight: float,
    dice_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Mask reductions cover tens of thousands of pixels. Keep the complete loss
    # path in FP32 even when the model forward pass uses autocast/FP16.
    image_logits = image_logits.float()
    heatmap_logits = heatmap_logits.float()
    labels = labels.float()
    masks = masks.float()
    mask_valid = mask_valid.float()
    image_loss = F.binary_cross_entropy_with_logits(image_logits, labels)
    supervised = mask_valid > 0.5
    zero = image_loss.new_zeros(())
    mask_bce = zero
    dice_loss = zero
    if supervised.any():
        pred = heatmap_logits[supervised]
        target = F.interpolate(masks[supervised], size=pred.shape[-2:], mode="nearest")
        positives = target.sum(dtype=torch.float32).clamp_min(1.0)
        negatives = (float(target.numel()) - positives).clamp_min(1.0)
        pos_weight = (negatives / positives).clamp(1.0, 20.0)
        mask_bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        prob = torch.sigmoid(pred)
        intersection = (prob * target).sum(dim=(1, 2, 3))
        denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = image_weight * image_loss + mask_bce_weight * mask_bce + dice_weight * dice_loss
    return total, {"image_bce": image_loss, "mask_bce": mask_bce, "dice_loss": dice_loss}


@torch.no_grad()
def localization_batch_metrics(heatmap_logits: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = F.interpolate(masks, size=heatmap_logits.shape[-2:], mode="nearest")
    prob = torch.sigmoid(heatmap_logits)
    flat_index = prob.flatten(1).argmax(dim=1)
    target_flat = target.flatten(1)
    point_hit = target_flat.gather(1, flat_index[:, None]).flatten()
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    soft_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return point_hit, soft_dice
