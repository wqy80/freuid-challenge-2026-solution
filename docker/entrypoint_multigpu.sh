#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/submissions}"
WORK_DIR="$OUTPUT_DIR/.freuid_work_multigpu"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
WORKERS="${WORKERS:-4}"
CONVNEXT_BATCH="${CONVNEXT_BATCH:-8}"
SWIN_BATCH="${SWIN_BATCH:-8}"
LOCALIZATION_BATCH="${LOCALIZATION_BATCH:-16}"
FINAL_CANDIDATE="${FINAL_CANDIDATE:-private_localization}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 4 ]]; then
  echo "ERROR: entrypoint_multigpu.sh requires four GPU IDs" >&2
  exit 2
fi
if ! compgen -G "$DATA_DIR/*.jpeg" >/dev/null; then
  echo "ERROR: no .jpeg images found in $DATA_DIR" >&2
  exit 3
fi

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/submission.csv" "$OUTPUT_DIR"/private_*.csv

run_fullimage() {
  local gpu="$1" checkpoint="$2" output="$3" batch="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python /app/scripts/infer.py \
    --checkpoint "$checkpoint" --test-dir "$DATA_DIR" --out "$output" \
    --batch-size "$batch" --workers "$WORKERS" --tta hflip
}

conv_folds=(0 1 2 4)
pids=()
for index in 0 1 2 3; do
  fold="${conv_folds[$index]}"
  run_fullimage "${GPUS[$index]}" \
    "/app/weights/convnext_large_fold${fold}.pt" \
    "$WORK_DIR/convnext_fold${fold}.csv" "$CONVNEXT_BATCH" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

python /app/scripts/ensemble_preds.py \
  --pred "$WORK_DIR/convnext_fold0.csv" \
  --pred "$WORK_DIR/convnext_fold1.csv" \
  --pred "$WORK_DIR/convnext_fold2.csv" \
  --pred "$WORK_DIR/convnext_fold4.csv" \
  --out "$WORK_DIR/convnext.csv"

pids=()
for fold in 0 1 2; do
  run_fullimage "${GPUS[$fold]}" \
    "/app/weights/swinv2_large_fold${fold}.pt" \
    "$WORK_DIR/swinv2_fold${fold}.csv" "$SWIN_BATCH" &
  pids+=("$!")
done
CUDA_VISIBLE_DEVICES="${GPUS[3]}" python /app/scripts/infer_synthetic_tamper_localization.py \
  --checkpoint /app/weights/synthetic_localization_fold2_epoch3.pt \
  --test-dir "$DATA_DIR" --out "$WORK_DIR/localization.csv" \
  --batch-size "$LOCALIZATION_BATCH" --workers "$WORKERS" --tta none &
pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

python /app/scripts/ensemble_preds.py \
  --pred "$WORK_DIR/swinv2_fold0.csv" \
  --pred "$WORK_DIR/swinv2_fold1.csv" \
  --pred "$WORK_DIR/swinv2_fold2.csv" \
  --out "$WORK_DIR/swinv2.csv"

PARTIAL_SELECTED="$WORK_DIR/selected_partial.csv"
python /app/scripts/make_private_candidates.py \
  --convnext-pred "$WORK_DIR/convnext.csv" \
  --swinv2-pred "$WORK_DIR/swinv2.csv" \
  --localization-pred "$WORK_DIR/localization.csv" \
  --test-dir "$DATA_DIR" --config /app/configs/private_candidates.json \
  --out-dir "$OUTPUT_DIR" --select "$FINAL_CANDIDATE" \
  --selected-out "$PARTIAL_SELECTED"

for candidate in private_base private_resolution_safe private_localization private_resolution_localization; do
  python /app/scripts/validate_container_output.py \
    --image-dir "$DATA_DIR" --submission "$OUTPUT_DIR/${candidate}.csv"
done

cp "$PARTIAL_SELECTED" "$OUTPUT_DIR/submission.csv"

rm -rf "$WORK_DIR"
echo "selected $FINAL_CANDIDATE -> $OUTPUT_DIR/submission.csv"
