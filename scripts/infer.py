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
from freuid.modeling import build_model
from freuid.paths import DEFAULT_PUBLIC_TEST_DIR
from freuid.utils import ensure_dir


@torch.no_grad()
def predict_one(model, loader, device, amp, tta):
    model.eval()
    ids, preds = [], []
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images).flatten()
            prob = torch.sigmoid(logits.float())
            if tta == "hflip":
                logits_flip = model(torch.flip(images, dims=[3])).flatten()
                prob = 0.5 * (prob + torch.sigmoid(logits_flip.float()))
        ids.extend(batch["id"])
        preds.extend(prob.detach().cpu().numpy().tolist())
    return pd.DataFrame({"id": ids, "label": preds})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=str(DEFAULT_PUBLIC_TEST_DIR))
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out", default="outputs/pred_public_test.csv")
    parser.add_argument("--model", default=None)
    parser.add_argument("--image-size", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir)
    ids = sorted(p.stem for p in test_dir.glob("*.jpeg"))
    if not ids:
        raise FileNotFoundError(f"no jpeg files found in {test_dir}")

    all_preds = []
    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = ckpt.get("args", {})
        model_name = args.model or ckpt_args.get("model")
        image_size = args.image_size or ckpt_args.get("image_size", "640,1024")
        if not model_name:
            raise ValueError("model name missing; pass --model")
        dataset = FreuidTestDataset(ids, test_dir, image_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        model = build_model(model_name, pretrained=False, image_size=image_size).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        pred = predict_one(model, loader, device, amp=(not args.no_amp and torch.cuda.is_available()), tta=args.tta)
        all_preds.append(pred["label"].values)

    out = pd.DataFrame({"id": ids, "label": np.mean(np.vstack(all_preds), axis=0)})
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()
