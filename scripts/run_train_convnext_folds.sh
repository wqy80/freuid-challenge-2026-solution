#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
TRAIN_CSV="${TRAIN_CSV:-data/train_labels_folds.csv}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
BASE_PORT="${BASE_PORT:-29740}"

for fold in 0 1 2 4; do
  out="outputs/convnext_large_992x1568_fold${fold}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_CACHE="$HF_HUB_CACHE" HF_HUB_OFFLINE=1 \
    torchrun --nproc_per_node="$NPROC" --master_addr=127.0.0.1 \
    --master_port="$((BASE_PORT + fold))" scripts/train.py \
    --data-root "$DATA_ROOT" --train-csv "$TRAIN_CSV" \
    --fold "$fold" --fold-column fold_stratified \
    --model convnext_large.fb_in22k_ft_in1k \
    --image-size 992,1568 --epochs 8 --batch-size 6 \
    --valid-batch-size 8 --workers 8 --lr 5e-5 \
    --weight-decay 0.05 --dist-timeout-minutes 180 \
    --out-dir "$out" 2>&1 | tee -a "$out/train.log"
done
