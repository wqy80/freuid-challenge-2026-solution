#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
PORT="${PORT:-29996}"
OUT="${OUT:-outputs/synthetic_tamper_localization/convnext_base_672x1056_fold2_mask_v1}"
DATA_ROOT="${DATA_ROOT:-data}"
TRAIN_CSV="${TRAIN_CSV:-data/train_labels_folds.csv}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_CACHE="$HF_HUB_CACHE" HF_HUB_OFFLINE=1 torchrun \
  --nproc_per_node="$NPROC" \
  --master_addr=127.0.0.1 \
  --master_port="$PORT" \
  scripts/train_synthetic_tamper_localization.py \
  --data-root "$DATA_ROOT" \
  --train-csv "$TRAIN_CSV" \
  --out-dir "$OUT" \
  --model convnext_base.fb_in22k_ft_in1k \
  --image-size 672,1056 \
  --fold 2 \
  --fold-column fold_stratified \
  --epochs 4 \
  --batch-size 4 \
  --valid-batch-size 8 \
  --workers 6 \
  --lr 5e-5 \
  --synth-probability 0.35 \
  --synthetic-valid-samples 1200 \
  --image-loss-weight 1.0 \
  --mask-bce-weight 1.0 \
  --dice-weight 1.0 \
  --amp-dtype bf16 \
  --dist-timeout-minutes 180 \
  2>&1 | tee -a "$OUT/train.log"
