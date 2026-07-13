# Model Weights

Put these eight files in this folder:

```text
convnext_large_fold0.pt
convnext_large_fold1.pt
convnext_large_fold2.pt
convnext_large_fold4.pt
swinv2_large_fold0.pt
swinv2_large_fold1.pt
swinv2_large_fold2.pt
synthetic_localization_fold2_epoch3.pt
```

The expected sizes and SHA-256 hashes are in `MANIFEST.csv`. After downloading
the GitHub Release assets, check them with:

```bash
python scripts/check_release_weights.py --weights-dir weights
```

The `.pt` files are GitHub Release assets and are ignored by Git. This avoids a
5.8 GB Git history while keeping the exact checkpoints publicly downloadable.
