# Frozen ResNet-18 features plus ridge

The fixed ImageNet ResNet-18 encoder replaces 13 hand-written measurements with
512 pooled image features. Encoder weights stay frozen, batch normalization
stays in evaluation mode, and feature extraction runs on CPU under inference
mode. Ridge still uses alpha 10, equal photo averaging, and training-fold-only
scaling. Settings were committed in `61dfacb` before feature extraction.

| Model | Local EMD | Paired-camera disagreement EMD |
| --- | ---: | ---: |
| Linear ridge on original 13 features | 41.2059 | 28.8910 |
| Submitted grayscale multi-crop blend | 40.1178 | 31.5259 |
| **Frozen ResNet-18 + ridge** | **49.6530** | **46.9890** |

Lower is better. All 24 physical soils participate in leave-one-soil-out
validation, with every camera view of the held-out soil excluded from ridge
training. The camera diagnostic covers the same 21 paired soils. Computing
features once is safe here because the encoder is fixed and does not update
parameters or normalization statistics from these photos.

Compared with the submitted blend, this model improves six soils and worsens
18. F827 improves by 89.68 EMD and H493 by 43.14, but G190 worsens by 106.38.
Compared with linear ridge on the original features, nine soils improve and
15 worsen. All ten test predictions change.

**Decision: retain the existing Kaggle submission.** Both predeclared local
criteria fail, so this candidate was not submitted. The mixed per-soil results
suggest the learned features contain useful information, but this fixed encoder
and ridge setting is not a better overall model. One configuration does not
rule out pretrained features generally.

Before fine-tuning an encoder, the next useful comparison is ridge regularization
selected inside each outer training fold, using the cached frozen features.
There are 512 features but only 23 training soils per outer fold. A nested,
soil-grouped comparison can test whether the fixed alpha was unsuitable without
selecting it on the outer held-out label or the public leaderboard. Keep the
encoder and crops fixed and declare the small candidate grid before running it.

## Inputs and reproducibility

The official [competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules)
allow external models subject to public-access and applicable license conditions
(§2.6 and §2.5.a.3). An authenticated API read of all seven published competition
pages found no contrary restriction. Rules receipts remain under
`artifacts/reports/pretrained_rules_*_2026-09-10.json`.

The encoder uses explicit
[ResNet18_Weights.IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
weights. The full checkpoint SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Preprocessing preserves the original calibrated 100 mm square resized to 256
pixels with Lanczos, then applies ImageNet RGB normalization. It deliberately
omits the published recipe's additional 224-pixel crop to preserve physical
coverage.

This run used torch 2.14.0, torchvision 0.29.0, NumPy 2.4.6, and Pillow 12.2.0.
Vision packages are optional; the lockfile records resolved versions.

```bash
uv sync --extra dev --extra vision
make frozen-model
make validate SUBMISSION=artifacts/submissions/frozen_resnet18_ridge.csv
make test
```

The first run downloads the 44.7 MB official checkpoint into the local torch
cache. Outputs under `artifacts/experiments/frozen_resnet18/` include 162 photo
feature rows, held-out predictions, camera predictions, per-soil comparisons,
and a summary recording package versions and hashes of the checkpoint, metadata,
labels, template, references, and all 162 source photos. Generated outputs stay
out of Git.

All 52 local tests pass. New tests verify frozen/evaluation/inference behavior,
unchanged batch-normalization state, batch independence, RGB normalization,
full crop coverage, checkpoint integrity, and soil alignment. CI separately
checks the optional CPU vision path using fixture weights, without downloading
pretrained weights. Rerunning the original image model reproduces its five
artifacts exactly; all 56 earlier artifacts remain byte-for-byte unchanged.
