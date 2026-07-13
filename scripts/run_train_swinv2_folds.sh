#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
TRAIN_CSV="${TRAIN_CSV:-data/train_labels_folds.csv}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
BASE_PORT="${BASE_PORT:-29750}"

for fold in 0 1 2; do
  out="outputs/swinv2_large_768x1152_fold${fold}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_CACHE="$HF_HUB_CACHE" HF_HUB_OFFLINE=1 \
    torchrun --nproc_per_node="$NPROC" --master_addr=127.0.0.1 \
    --master_port="$((BASE_PORT + fold))" scripts/train.py \
    --data-root "$DATA_ROOT" --train-csv "$TRAIN_CSV" \
    --fold "$fold" --fold-column fold_stratified \
    --model swinv2_large_window12_192 \
    --image-size 768,1152 --epochs 8 --batch-size 5 \
    --valid-batch-size 8 --workers 4 --lr 3e-5 \
    --weight-decay 0.05 --dist-timeout-minutes 180 \
    --out-dir "$out" 2>&1 | tee -a "$out/train.log"
done
