from __future__ import annotations

import torch


def _parse_size(image_size: str | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if image_size is None:
        return None
    if isinstance(image_size, str):
        parts = image_size.lower().replace("x", ",").split(",")
        if len(parts) != 2:
            raise ValueError("image_size must look like 640,1024 or 640x1024")
        return int(parts[0]), int(parts[1])
    return int(image_size[0]), int(image_size[1])


def build_model(model_name: str, pretrained: bool = True, in_chans: int = 3, image_size=None) -> torch.nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required. Install with: pip install timm") from exc

    kwargs = {"pretrained": pretrained, "in_chans": in_chans, "num_classes": 1}
    parsed_size = _parse_size(image_size)
    if parsed_size is not None:
        kwargs["img_size"] = parsed_size
    try:
        return timm.create_model(model_name, **kwargs)
    except TypeError:
        kwargs.pop("img_size", None)
        return timm.create_model(model_name, **kwargs)
