from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freuid.data import FreuidTestDataset
from freuid.synthetic_tamper import ConvNeXtFpnLocalization
from freuid.utils import ensure_dir


@torch.no_grad()
def predict(model, loader, device, amp, amp_dtype, tta):
    model.eval()
    ids, predictions = [], []
    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            logits, _ = model(image)
            probability = torch.sigmoid(logits.float())
            if tta == "hflip":
                flip_logits, _ = model(torch.flip(image, dims=[3]))
                probability = 0.5 * (probability + torch.sigmoid(flip_logits.float()))
        ids.extend(batch["id"])
        predictions.extend(probability.cpu().numpy().tolist())
    return ids, np.asarray(predictions, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tta", choices=["none", "hflip"], default="hflip")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    test_dir = Path(args.test_dir)
    ids = sorted(path.stem for path in test_dir.glob("*.jpeg"))
    if not ids:
        raise FileNotFoundError(f"no JPEG files found in {test_dir}")

    all_predictions = []
    expected_ids = None
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_args = checkpoint.get("args", {})
        image_size = checkpoint_args.get("image_size", "672,1056")
        dataset = FreuidTestDataset(ids, test_dir, image_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        model = ConvNeXtFpnLocalization(
            model_name=checkpoint_args.get("model", "convnext_base.fb_in22k_ft_in1k"),
            pretrained=False,
            fpn_channels=int(checkpoint_args.get("fpn_channels", 128)),
            topk_fraction=float(checkpoint_args.get("topk_fraction", 0.01)),
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_ids, predictions = predict(model, loader, device, amp, amp_dtype, args.tta)
        if expected_ids is None:
            expected_ids = checkpoint_ids
        elif checkpoint_ids != expected_ids:
            raise RuntimeError("checkpoint inference ID order mismatch")
        all_predictions.append(predictions)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = pd.DataFrame({"id": expected_ids, "label": np.mean(np.stack(all_predictions), axis=0)})
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    output.to_csv(out_path, index=False)
    print(
        f"wrote {out_path} rows={len(output)} checkpoints={len(all_predictions)} "
        f"min={output.label.min():.8g} max={output.label.max():.8g}"
    )


if __name__ == "__main__":
    main()
