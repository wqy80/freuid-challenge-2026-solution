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

from freuid.data import FreuidTrainDataset
from freuid.metrics import freuid_score
from freuid.modeling import build_model
from freuid.paths import DEFAULT_DATA_ROOT, DEFAULT_TRAIN_CSV
from freuid.splits import add_folds
from freuid.utils import ensure_dir, is_rank0, seed_everything, unwrap_model


def setup_distributed(timeout_minutes: int):
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=timeout_minutes))
    return True, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


def make_loader(df, data_root, image_size, mode, batch_size, workers, distributed):
    dataset = FreuidTrainDataset(df, data_root, image_size, mode)
    sampler = DistributedSampler(dataset, shuffle=(mode == "train")) if distributed and mode == "train" else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and mode == "train"),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=(mode == "train"),
        persistent_workers=workers > 0,
    ), sampler


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum, clip_grad, amp):
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, disable=not is_rank0(), leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images).flatten()
            loss = criterion(logits, labels) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu()) * grad_accum
        losses.append(loss_value)
        pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
    return float(np.mean(losses))


def reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    if not distributed:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


@torch.no_grad()
def validate(model, loader, device, amp, distributed):
    model.eval()
    ids, labels, preds = [], [], []
    for batch in tqdm(loader, disable=not is_rank0(), leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images).flatten()
        prob = torch.sigmoid(logits.float()).detach().cpu().numpy()
        preds.extend(prob.tolist())
        labels.extend(batch["label"].numpy().tolist())
        ids.extend(batch["id"])
    local_oof = pd.DataFrame({"id": ids, "label": labels, "pred": preds})
    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_oof)
        if not is_rank0():
            return None, None
        oof = pd.concat(gathered, ignore_index=True)
    else:
        oof = local_oof
    metrics = freuid_score(oof["label"].to_numpy(), oof["pred"].to_numpy())
    return metrics, oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--out-dir", default="outputs/convnext_base_640")
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--image-size", default="640,1024")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fold-column", default="fold_stratified")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--valid-batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dist-timeout-minutes", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pos-weight", default="auto")
    args = parser.parse_args()

    distributed, rank, local_rank, world_size = setup_distributed(args.dist_timeout_minutes)
    seed_everything(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.out_dir)

    df = pd.read_csv(args.train_csv)
    if args.fold_column not in df.columns:
        df = add_folds(df, args.n_splits, args.seed, "stratified")
    train_df = df[df[args.fold_column] != args.fold].reset_index(drop=True)
    valid_df = df[df[args.fold_column] == args.fold].reset_index(drop=True)

    if args.pos_weight == "auto":
        n_pos = max(float((train_df["label"].astype(int) == 1).sum()), 1.0)
        n_neg = max(float((train_df["label"].astype(int) == 0).sum()), 1.0)
        pos_weight = n_neg / n_pos
    else:
        pos_weight = float(args.pos_weight)

    if is_rank0():
        ensure_dir(out_dir)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"train={len(train_df)} valid={len(valid_df)} pos_weight={pos_weight:.4f}")

    train_loader, train_sampler = make_loader(train_df, args.data_root, args.image_size, "train", args.batch_size, args.workers, distributed)
    valid_batch_size = args.valid_batch_size if args.valid_batch_size > 0 else args.batch_size
    valid_eval_df = valid_df.iloc[rank::world_size].reset_index(drop=True) if distributed else valid_df
    valid_loader, _ = make_loader(valid_eval_df, args.data_root, args.image_size, "valid", valid_batch_size, args.workers, False)

    model = build_model(args.model, pretrained=not args.no_pretrained, image_size=args.image_size).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp and torch.cuda.is_available())
    amp = not args.no_amp and torch.cuda.is_available()

    best = float("inf")
    for epoch in range(args.epochs):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.grad_accum, args.clip_grad, amp)
        train_loss = reduce_mean(train_loss, device, distributed)
        scheduler.step()
        if distributed:
            dist.barrier()

        metrics, oof = validate(model, valid_loader, device, amp, distributed)
        if is_rank0():
            metrics["train_loss"] = train_loss
            metrics["epoch"] = epoch
            metrics["seconds"] = round(time.time() - start, 1)
            print(json.dumps(metrics, indent=2))
            oof.to_csv(out_dir / f"oof_fold{args.fold}.csv", index=False)
            torch.save({"model": unwrap_model(model).state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "last.pt")
            if metrics["freuid"] < best:
                best = metrics["freuid"]
                torch.save({"model": unwrap_model(model).state_dict(), "args": vars(args), "metrics": metrics}, out_dir / "best.pt")
        if distributed:
            dist.barrier()

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
