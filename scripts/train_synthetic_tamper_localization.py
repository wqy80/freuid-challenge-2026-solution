from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freuid.metrics import freuid_score
from freuid.synthetic_tamper import (
    ConvNeXtFpnLocalization,
    SyntheticTamperConfig,
    SyntheticTamperDataset,
    localization_batch_metrics,
    localization_loss,
)
from freuid.synthetic_tamper_v2 import SyntheticTamperV2Config, SyntheticTamperV2Dataset
from freuid.synthetic_tamper_v3 import SyntheticTamperV3Config, SyntheticTamperV3Dataset
from freuid.utils import ensure_dir, is_rank0, seed_everything, unwrap_model


def setup_distributed(timeout_minutes: int):
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=timeout_minutes))
    return True, rank, local_rank, world_size


def reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def make_loader(dataset, batch_size, workers, distributed, train):
    sampler = DistributedSampler(dataset, shuffle=train) if distributed and train else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and train),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def train_epoch(model, loader, optimizer, scaler, device, amp, amp_dtype, args):
    model.train()
    totals = {"loss": [], "image_bce": [], "mask_bce": [], "dice_loss": [], "synthetic_fraction": []}
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, disable=not is_rank0(), leave=False)
    for step, batch in enumerate(progress):
        image = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        mask_valid = batch["mask_valid"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            image_logits, heatmap_logits = model(image)
            loss, parts = localization_loss(
                image_logits,
                heatmap_logits,
                labels,
                masks,
                mask_valid,
                args.image_loss_weight,
                args.mask_bce_weight,
                args.dice_weight,
            )
            if not torch.isfinite(loss):
                values = {key: float(value.detach().float().cpu()) for key, value in parts.items()}
                diagnostics = {
                    "ids": list(batch["id"]),
                    "operations": list(batch["operation"]),
                    "image_finite": bool(torch.isfinite(image).all().item()),
                    "image_logits_finite": bool(torch.isfinite(image_logits).all().item()),
                    "heatmap_logits_finite": bool(torch.isfinite(heatmap_logits).all().item()),
                    "image_logits_min": float(torch.nan_to_num(image_logits.float()).min().detach().cpu()),
                    "image_logits_max": float(torch.nan_to_num(image_logits.float()).max().detach().cpu()),
                }
                raise FloatingPointError(
                    f"non-finite localization loss: total={float(loss.detach().float().cpu())}, "
                    f"parts={values}, diagnostics={diagnostics}"
                )
            scaled_loss = loss / args.grad_accum
        scaler.scale(scaled_loss).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        totals["loss"].append(float(loss.detach().cpu()))
        for key in ("image_bce", "mask_bce", "dice_loss"):
            totals[key].append(float(parts[key].detach().cpu()))
        totals["synthetic_fraction"].append(float(mask_valid.mean().detach().cpu()))
        progress.set_postfix(loss=f"{np.mean(totals['loss'][-40:]):.4f}", synth=f"{np.mean(totals['synthetic_fraction'][-40:]):.2f}")
    return {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}


@torch.no_grad()
def validate_oof(model, loader, device, amp, amp_dtype, distributed):
    model.eval()
    ids, labels, preds = [], [], []
    for batch in tqdm(loader, disable=not is_rank0(), leave=False):
        image = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            logits, _ = model(image)
        ids.extend(batch["id"])
        labels.extend(batch["label"].numpy().tolist())
        preds.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    local = pd.DataFrame({"id": ids, "label": labels, "pred": preds})
    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        if not is_rank0():
            return None, None
        local = pd.concat(gathered, ignore_index=True)
    return freuid_score(local["label"].to_numpy(), local["pred"].to_numpy()), local


@torch.no_grad()
def validate_synthetic(model, loader, device, amp, amp_dtype, distributed):
    model.eval()
    rows = []
    for batch in tqdm(loader, disable=not is_rank0(), leave=False):
        image = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            logits, heatmap_logits = model(image)
        point_hit, soft_dice = localization_batch_metrics(heatmap_logits.float(), masks)
        scores = torch.sigmoid(logits.float()).cpu().numpy()
        for index, image_id in enumerate(batch["id"]):
            rows.append(
                {
                    "id": image_id,
                    "operation": batch["operation"][index],
                    "pred": float(scores[index]),
                    "point_hit": float(point_hit[index].cpu()),
                    "soft_dice": float(soft_dice[index].cpu()),
                }
            )
    local = pd.DataFrame(rows)
    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        if not is_rank0():
            return None, None
        local = pd.concat(gathered, ignore_index=True)
    metrics = {
        "synthetic_rows": int(len(local)),
        "synthetic_score_mean": float(local["pred"].mean()),
        "point_hit": float(local["point_hit"].mean()),
        "soft_dice": float(local["soft_dice"].mean()),
    }
    metrics["localization_gate"] = 0.7 * metrics["point_hit"] + 0.3 * metrics["soft_dice"]
    return metrics, local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--image-size", default="672,1056")
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--fold-column", default="fold_stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--valid-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dist-timeout-minutes", type=int, default=180)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--synth-probability", type=float, default=0.35)
    parser.add_argument("--generator-version", choices=["v1", "v2", "v3"], default="v1")
    parser.add_argument("--recapture-probability", type=float, default=0.50)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--synthetic-valid-samples", type=int, default=1200)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--image-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--min-point-hit", type=float, default=0.95)
    parser.add_argument("--min-soft-dice", type=float, default=0.90)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    args = parser.parse_args()

    distributed, rank, local_rank, world_size = setup_distributed(args.dist_timeout_minutes)
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    out_dir = ensure_dir(args.out_dir)
    frame = pd.read_csv(args.train_csv)
    train_df = frame[frame[args.fold_column] != args.fold].reset_index(drop=True)
    valid_df = frame[frame[args.fold_column] == args.fold].reset_index(drop=True)
    image_size = tuple(int(value) for value in args.image_size.lower().replace("x", ",").split(","))
    if args.generator_version == "v3":
        config = SyntheticTamperV3Config(
            image_size=image_size,
            synth_probability=args.synth_probability,
            recapture_probability=args.recapture_probability,
            seed=args.seed,
        )
        dataset_class = SyntheticTamperV3Dataset
    elif args.generator_version == "v2":
        config = SyntheticTamperV2Config(image_size=image_size, synth_probability=args.synth_probability, seed=args.seed)
        dataset_class = SyntheticTamperV2Dataset
    else:
        config = SyntheticTamperConfig(image_size=image_size, synth_probability=args.synth_probability, seed=args.seed)
        dataset_class = SyntheticTamperDataset

    train_dataset = dataset_class(train_df, args.data_root, config, "train")
    valid_rank_df = valid_df.iloc[rank::world_size].reset_index(drop=True) if distributed else valid_df
    valid_dataset = dataset_class(valid_rank_df, args.data_root, config, "valid")
    synthetic_valid_df = valid_df[valid_df["label"].astype(int) == 0].sample(
        min(args.synthetic_valid_samples, int((valid_df["label"].astype(int) == 0).sum())), random_state=args.seed
    ).reset_index(drop=True)
    synthetic_valid_rank_df = synthetic_valid_df.iloc[rank::world_size].reset_index(drop=True) if distributed else synthetic_valid_df
    synthetic_valid_dataset = dataset_class(synthetic_valid_rank_df, args.data_root, config, "synthetic_valid")

    train_loader, train_sampler = make_loader(train_dataset, args.batch_size, args.workers, distributed, True)
    valid_loader, _ = make_loader(valid_dataset, args.valid_batch_size, args.workers, False, False)
    synthetic_valid_loader, _ = make_loader(synthetic_valid_dataset, args.valid_batch_size, args.workers, False, False)

    model = ConvNeXtFpnLocalization(
        args.model,
        pretrained=not args.no_pretrained,
        fpn_channels=args.fpn_channels,
        topk_fraction=args.topk_fraction,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(initial["model"], strict=True)
        if is_rank0():
            print(f"initialized from {args.init_checkpoint}")
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)

    if is_rank0():
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(json.dumps({"train": len(train_df), "valid": len(valid_df), "synthetic_valid": len(synthetic_valid_df), "fold": args.fold}, indent=2))

    best_freuid = float("inf")
    best_gate = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, amp, amp_dtype, args)
        train_metrics = {key: reduce_mean(value, device, distributed) for key, value in train_metrics.items()}
        scheduler.step()
        if distributed:
            dist.barrier()
        oof_metrics, oof = validate_oof(model, valid_loader, device, amp, amp_dtype, distributed)
        synthetic_metrics, synthetic_rows = validate_synthetic(model, synthetic_valid_loader, device, amp, amp_dtype, distributed)
        if is_rank0():
            metrics = {**oof_metrics, **synthetic_metrics, **{f"train_{key}": value for key, value in train_metrics.items()}}
            metrics.update({"epoch": epoch, "seconds": round(time.time() - start, 1)})
            print(json.dumps(metrics, indent=2))
            oof["fold"] = args.fold
            oof.to_csv(out_dir / f"oof_fold{args.fold}_epoch{epoch}.csv", index=False)
            synthetic_rows.to_csv(out_dir / f"synthetic_valid_epoch{epoch}.csv", index=False)
            state = {"model": unwrap_model(model).state_dict(), "args": vars(args), "metrics": metrics}
            torch.save(state, out_dir / f"epoch{epoch}.pt")
            torch.save(state, out_dir / "last.pt")
            localization_ok = (
                metrics["point_hit"] >= args.min_point_hit
                and metrics["soft_dice"] >= args.min_soft_dice
            )
            better_oof = metrics["freuid"] < best_freuid - 1e-12
            tied_oof_better_localization = (
                abs(metrics["freuid"] - best_freuid) <= 1e-12
                and metrics["localization_gate"] > best_gate
            )
            if localization_ok and (better_oof or tied_oof_better_localization):
                best_freuid = metrics["freuid"]
                best_gate = metrics["localization_gate"]
                best_epoch = epoch
                torch.save(state, out_dir / "best.pt")
                oof.to_csv(out_dir / f"best_oof_fold{args.fold}.csv", index=False)
                synthetic_rows.to_csv(out_dir / "best_synthetic_valid.csv", index=False)
            (out_dir / "best_epoch.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "best_freuid": best_freuid,
                        "best_localization_gate": best_gate,
                        "min_point_hit": args.min_point_hit,
                        "min_soft_dice": args.min_soft_dice,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
