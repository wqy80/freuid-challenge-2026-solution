# FREUID Challenge 2026

This repository contains our frozen training and inference code for the FREUID
Challenge 2026. Competition images are not included.

## Weights

Download the checkpoint files from the GitHub Release and place them in
`weights/`. Check their sizes and hashes with:

```bash
python scripts/check_release_weights.py --weights-dir weights
```

The expected SHA-256 values are in `weights/MANIFEST.csv`.

## Docker inference

Build the image:

```bash
docker build -t freuid-final:latest .
```

Run it without network access:

```bash
docker run --rm --gpus all --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-final:latest
```

The result is written to:

```text
/submissions/submission.csv
```

It contains one `id,label` row for every JPEG file mounted at `/data`.

For a four-GPU machine, use:

```bash
docker run --rm --gpus all --network none \
  --entrypoint /app/docker/entrypoint_multigpu.sh \
  -e GPU_LIST=0,1,2,3 \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-final:latest
```

## Training

Create the folds with `scripts/make_folds.py`, then use:

```text
scripts/run_train_convnext_folds.sh
scripts/run_train_swinv2_folds.sh
scripts/run_synthetic_tamper_fold2.sh
```

The scripts contain the frozen model names, image sizes, folds, and training
arguments. Starting checkpoints are public ImageNet weights from `timm`.

## Environment

Package versions for inference are pinned in `requirements-inference.txt`.
Training dependencies are listed in `requirements-training.txt`. Hardware and
runtime notes are in `ENVIRONMENT.md`.

## License

The source code is released under the MIT License. Competition data and
pretrained checkpoints keep their original licenses.
