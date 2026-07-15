# FREUID Challenge 2026

This repository contains our frozen solution for the FREUID Challenge 2026.
It includes the training code, inference code, Docker file, model settings, and
weight hashes. Competition images are not included.

- Team: `tianboguangding`
- Frozen code tag: `code-freeze-20260713-r3`
- Frozen commit: `cba35e523bbb2d987f1a1dd70ceef0ddddd3cde5`
- Best public score: `0.00052`

The short report is available as
[`reports/technical_report.pdf`](reports/technical_report.pdf). The Markdown
version is in the same folder.

## Method

The final system has three parts:

| Part | Model | Input size | Folds | TTA |
| --- | --- | ---: | --- | --- |
| Full image 1 | ConvNeXt-L | 992 x 1568 | 0, 1, 2, 4 | horizontal flip |
| Full image 2 | SwinV2-L | 768 x 1152 | 0, 1, 2 | horizontal flip |
| Local edit | ConvNeXt-B + FPN | 672 x 1056 | 2, epoch 3 | none |

We first average predictions between folds. ConvNeXt and SwinV2 are mixed with
weights `1.0` and `0.8`. The local-edit model is then added with a small weight
of `0.003`. A higher output score means that the document is more likely to be
an attack.

The full-image system scored `0.00063` on the public leaderboard. The small
local-edit branch improved it to `0.00052`.

## Weights

Download the eight checkpoint files from the
[GitHub Release](https://github.com/wqy80/freuid-challenge-2026-solution/releases/tag/code-freeze-20260713)
and place them in `weights/`.

Check every file before inference:

```bash
python scripts/check_release_weights.py --weights-dir weights
```

Expected file sizes and SHA-256 hashes are stored in
`weights/MANIFEST.csv`.

## Input data

The input folder must be flat. Put all JPEG files directly in one folder:

```text
/data/
  document_id_1.jpeg
  document_id_2.jpeg
  ...
```

The file name without `.jpeg` is used as the submission `id`.

## External data

We did not use external document images or private labels. The full-image
backbones start from public ImageNet-pretrained weights available through
`timm`. Synthetic local edits are made only from competition training images.

## Docker inference

Build the image:

```bash
docker build -t freuid-final:latest .
```

Run on one GPU:

```bash
mkdir -p out

docker run --rm --gpus all --network none \
  -e GPU=0 \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-final:latest
```

Run on four GPUs:

```bash
mkdir -p out

docker run --rm --gpus all --network none --shm-size=32g \
  --entrypoint /app/docker/entrypoint_multigpu.sh \
  -e GPU_LIST=0,1,2,3 \
  -e WORKERS=4 \
  -e CONVNEXT_BATCH=8 \
  -e SWIN_BATCH=8 \
  -e LOCALIZATION_BATCH=16 \
  -e FINAL_CANDIDATE=private_localization \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-final:latest
```

The selected prediction is written to:

```text
out/submission.csv
```

The container also writes four frozen candidates:

```text
private_base.csv
private_resolution_safe.csv
private_localization.csv
private_resolution_localization.csv
```

Each output has the columns `id,label`. The container checks IDs, duplicate
rows, missing values, and the score range before it exits.

## Final submissions

We selected two Kaggle submissions:

- `private_localization_full.csv`
- `private_resolution_safe_full.csv`

Both files contain 142,818 rows: 7,821 public rows and 134,997 private rows.
They use the same frozen weights. The second file is a safer choice when the
test image-size distribution changes.

No model was trained or updated with private test images.

## Training

Create the five folds with:

```bash
python scripts/make_folds.py
```

The frozen training commands are in:

```text
scripts/run_train_convnext_folds.sh
scripts/run_train_swinv2_folds.sh
scripts/run_synthetic_tamper_fold2.sh
```

The starting weights are public ImageNet weights provided through `timm`.
The scripts contain the model names, image sizes, folds, and training options.

## Environment

Inference packages are listed in `requirements-inference.txt`. Training
packages are listed in `requirements-training.txt`. Training and final checks
used four NVIDIA A40 GPUs with 46 GB memory each. Full software details are in
[`ENVIRONMENT.md`](ENVIRONMENT.md).

## License

The source code uses the MIT License. Competition data and pretrained weights
keep their original licenses.
