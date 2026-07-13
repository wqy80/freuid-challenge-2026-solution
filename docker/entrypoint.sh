#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/submissions}"
WORK_DIR="$OUTPUT_DIR/.freuid_work"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"
CONVNEXT_BATCH="${CONVNEXT_BATCH:-8}"
SWIN_BATCH="${SWIN_BATCH:-8}"
LOCALIZATION_BATCH="${LOCALIZATION_BATCH:-16}"
FINAL_CANDIDATE="${FINAL_CANDIDATE:-private_localization}"

mkdir -p "$WORK_DIR"
mkdir -p "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
rm -f "$OUTPUT_DIR/submission.csv" \
  "$OUTPUT_DIR/private_base.csv" \
  "$OUTPUT_DIR/private_resolution_safe.csv" \
  "$OUTPUT_DIR/private_localization.csv" \
  "$OUTPUT_DIR/private_resolution_localization.csv" \
  "$OUTPUT_DIR/candidate_diagnostics.csv"

if ! compgen -G "$DATA_DIR/*.jpeg" >/dev/null; then
  echo "ERROR: no .jpeg images found in $DATA_DIR" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU" python /app/scripts/infer.py \
  --checkpoint /app/weights/convnext_large_fold0.pt \
  --checkpoint /app/weights/convnext_large_fold1.pt \
  --checkpoint /app/weights/convnext_large_fold2.pt \
  --checkpoint /app/weights/convnext_large_fold4.pt \
  --test-dir "$DATA_DIR" \
  --out "$WORK_DIR/convnext.csv" \
  --batch-size "$CONVNEXT_BATCH" \
  --workers "$WORKERS" \
  --tta hflip

CUDA_VISIBLE_DEVICES="$GPU" python /app/scripts/infer.py \
  --checkpoint /app/weights/swinv2_large_fold0.pt \
  --checkpoint /app/weights/swinv2_large_fold1.pt \
  --checkpoint /app/weights/swinv2_large_fold2.pt \
  --test-dir "$DATA_DIR" \
  --out "$WORK_DIR/swinv2.csv" \
  --batch-size "$SWIN_BATCH" \
  --workers "$WORKERS" \
  --tta hflip

CUDA_VISIBLE_DEVICES="$GPU" python /app/scripts/infer_synthetic_tamper_localization.py \
  --checkpoint /app/weights/synthetic_localization_fold2_epoch3.pt \
  --test-dir "$DATA_DIR" \
  --out "$WORK_DIR/localization.csv" \
  --batch-size "$LOCALIZATION_BATCH" \
  --workers "$WORKERS" \
  --tta none

python /app/scripts/make_private_candidates.py \
  --convnext-pred "$WORK_DIR/convnext.csv" \
  --swinv2-pred "$WORK_DIR/swinv2.csv" \
  --localization-pred "$WORK_DIR/localization.csv" \
  --test-dir "$DATA_DIR" \
  --config /app/configs/private_candidates.json \
  --out-dir "$OUTPUT_DIR" \
  --select "$FINAL_CANDIDATE" \
  --selected-out "$OUTPUT_DIR/submission.csv"

for candidate in \
  submission \
  private_base \
  private_resolution_safe \
  private_localization \
  private_resolution_localization; do
  python /app/scripts/validate_container_output.py \
    --image-dir "$DATA_DIR" \
    --submission "$OUTPUT_DIR/${candidate}.csv"
done

rm -rf "$WORK_DIR" "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
echo "selected $FINAL_CANDIDATE -> $OUTPUT_DIR/submission.csv"
