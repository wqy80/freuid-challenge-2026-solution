# FREUID Challenge 2026 - Technical Report

**Team:** tianboguangding

**Kaggle username:** tianboguangding

**Repository:** https://github.com/wqy80/freuid-challenge-2026-solution

**Frozen code tag:** `code-freeze-20260713-r3`

## 1. Short summary

Our solution is an ensemble of two full-image classifiers and one small
localization model. The main models are ConvNeXt-L and SwinV2-L. They use
different input sizes, so they do not see the document in exactly the same
way. The localization model is a ConvNeXt-B with an FPN head. It was trained
with simple synthetic local edits and is given only a very small weight in the
final score.

The full-image ensemble reached a public score of **0.00063**. Adding the
localization model with weight 0.003 improved the public score to **0.00052**.
All scores in this report are public leaderboard scores. We did not use public
test labels or pseudo-labels.

## 2. Data and split

The training set contains 69,352 document images:

- 40,005 bona-fide images
- 29,347 attack images

We made five stratified folds with seed 42. The split keeps the class balance
close between folds. The final full-image system uses ConvNeXt folds 0, 1, 2,
and 4, and SwinV2 folds 0, 1, and 2. Fold 3 was not included because it made
the public ensemble worse in our tests.

We did not use external document images, private labels, or test pseudo-labels.
The backbones start from public ImageNet-pretrained weights distributed through
`timm`. Synthetic edits use only competition training images.

The images have several common sizes and aspect ratios. We did not crop the
document into a fixed square. Each model resizes the full RGB image with
bicubic interpolation and ImageNet normalization.

## 3. Full-image models

### ConvNeXt-L

- timm model: `convnext_large.fb_in22k_ft_in1k`
- input size: 992 x 1568
- folds: 0, 1, 2, 4
- epochs: 8
- batch size per GPU: 6
- optimizer: AdamW
- learning rate: 5e-5
- weight decay: 0.05
- loss: weighted binary cross entropy
- test-time augmentation: original image and horizontal flip

### SwinV2-L

- timm model: `swinv2_large_window12_192`
- input size: 768 x 1152
- folds: 0, 1, 2
- epochs: 8
- batch size per GPU: 5
- optimizer: AdamW
- learning rate: 3e-5
- weight decay: 0.05
- loss: weighted binary cross entropy
- test-time augmentation: original image and horizontal flip

For both models, the class weight is calculated from the training part of the
fold. We use a cosine learning-rate schedule. Training augmentation is mild:
small color changes and occasional Gaussian blur. We found that strong image
damage often reduced the public result.

Predictions are first averaged between folds. ConvNeXt and SwinV2 are then
combined in probability space with weights 1.0 and 0.8.

## 4. Synthetic localization model

The third model tries to find a small suspicious region instead of only making
one global image decision.

- encoder: `convnext_base.fb_in22k_ft_in1k`
- head: four-level FPN and a one-channel heatmap
- input size: 672 x 1056
- fold: 2
- selected checkpoint: epoch 3
- training epochs: 4
- batch size per GPU: 4
- optimizer: AdamW
- learning rate: 5e-5
- precision: bfloat16
- test-time augmentation: none

During training, some genuine images are changed with a local synthetic edit.
The operations include copy-move, donor splice, text replacement, photo
replacement, local JPEG compression, and local blur/inpainting. The generated
mask gives supervision to the heatmap head. The loss is the sum of image BCE,
mask BCE, and Dice loss.

This branch is useful but less stable than the full-image ensemble. For that
reason its probability weight is only 0.003. The final public prediction is:

```text
(full_image_prediction + 0.003 * localization_prediction) / 1.003
```

## 5. Inference

For each checkpoint, inference produces a continuous fraud score in [0, 1].
Higher means more likely to be an attack.

The default Docker candidate is `private_localization`:

- Average four ConvNeXt-L folds with horizontal-flip TTA.
- Average three SwinV2-L folds with horizontal-flip TTA.
- Blend ConvNeXt and SwinV2 with weights 1.0 and 0.8.
- Add the localization prediction with weight 0.003.
- Write one `id,label` row for every JPEG image in `/data`.

The repository also creates a plain full-image candidate and two
resolution-aware safety candidates. They use the same frozen checkpoints. The
extra candidates are kept because the private set may have a different image
size distribution. They do not train or update any model on test images.

## 6. Public results

| System | Public score |
| --- | ---: |
| ConvNeXt-L + SwinV2-L | 0.00063 |
| ConvNeXt-L + SwinV2-L + localization (0.003) | 0.00052 |

The public set contains 7,821 images, so very small score changes can depend on
only a few difficult samples. We therefore kept the localization weight small
and kept the plain full-image model as a safe candidate.

## 7. Reproducibility

The frozen release contains eight checkpoint files:

- four ConvNeXt-L checkpoints
- three SwinV2-L checkpoints
- one synthetic-localization checkpoint

Their exact sizes and SHA-256 hashes are stored in `weights/MANIFEST.csv`.
The script below checks all files:

```bash
python scripts/check_release_weights.py --weights-dir weights
```

Build and run the Docker image with:

```bash
docker build -t freuid-final:latest .

docker run --rm --gpus all --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-final:latest
```

The output is `/submissions/submission.csv`. A four-GPU entry point is also
included for faster inference. The image does not need network access at run
time. The output validator checks the columns, IDs, row count, finite values,
and the [0, 1] score range.

## 8. Hardware and software

Training used four NVIDIA A40 GPUs. The main validated server environment used
Python 3.12, PyTorch 2.10 with CUDA 12.6, timm 1.0.26, NumPy 2.4.4, and Pillow
12.0. A clean public Docker build based on PyTorch 2.5.1 and CUDA 11.8 also
passed an end-to-end smoke test. More details are in `ENVIRONMENT.md`.

## 9. Limits

The public leaderboard covers only a small part of the final test set. The
training data also has a limited set of document layouts and image sizes. A
model can learn layout shortcuts, and a public improvement may not transfer to
new countries or print-and-capture images. We keep the plain ensemble and the
resolution-aware candidates for this reason.

## 10. Final submission record

Private inference was completed with the frozen code and weights. We selected
two final Kaggle submissions on 15 July 2026:

- `private_localization_full.csv`
- `private_resolution_safe_full.csv`

Each file has 142,818 rows. This includes 7,821 public rows and 134,997 private
rows. Both selected files showed the same public score of **0.00052** because
their public rows are the same. Their private rows use two different frozen
candidate rules.

The localization file is our main candidate. The resolution-safe file is the
backup for a change in image-size distribution. No model training, fine-tuning,
or weight update was done with private images.
