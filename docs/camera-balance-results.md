# Equal camera weighting

This comparison changes only the way photo features are averaged. Each camera
contributes its mean feature vector, then the soil receives the mean of those
camera vectors. A phone with more photos therefore receives the same total
weight as a phone with fewer photos. Apply this to both training and test soils.

The grayscale features, 50/100/150 mm crops, 3-neighbor predictor, equal crop
blend, and physical-soil validation folds are unchanged. This reuses the saved
photo features and existing grouped evaluator. The execution and submission
rules were committed in `b1b0f88` before running the comparison.

| Crop/model | Photo-weighted local EMD | Camera-weighted local EMD | Photo-weighted camera disagreement | Camera-weighted camera disagreement |
| --- | ---: | ---: | ---: | ---: |
| Gray 50 mm | 39.9198 | 41.6224 | 43.4828 | 46.7908 |
| Gray 100 mm | 42.9820 | 43.9303 | 33.0769 | 52.1206 |
| Gray 150 mm | 41.5983 | 45.7725 | 32.6688 | 36.0087 |
| **Equal crop blend** | **40.1178** | **42.4228** | **31.5259** | **38.3402** |

Lower is better. Local validation covers all 24 soils; disagreement covers the
21 soils with paired cameras. Every view of a held-out soil stays excluded from
training, and feature scaling uses only that fold's training soils.

The balanced blend improves eight soils, ties six, and worsens ten. F827 improves
by 23.50 EMD, but H405 worsens by 46.04. Reweighting training examples also changes
nearest-neighbor choices for other soils, so an improvement on the motivating
audit case does not imply an overall gain. Nine of ten test predictions change.

**Decision: retain the submitted photo-weighted blend.** The new candidate fails
both predeclared local criteria and was not submitted to Kaggle. The result
does not resolve the unseen-iPhone distribution gap; it shows that this specific
averaging change does not help our local accuracy or camera diagnostic. Kernel
ridge remains the next model comparison in the roadmap.

## Reproduce

After `make experiments` and `make multicrop`:

```bash
make camera-balance
make validate SUBMISSION=artifacts/submissions/camera_balanced_multicrop.csv
make test
```

Outputs under `artifacts/experiments/camera_balance/` include component and blend
OOF predictions, camera predictions, per-soil changes, and a summary with input
SHA-256 fingerprints. The separately named candidate respects the submission
template. All 44 previous artifacts, including the submitted file and receipt,
remain byte-for-byte unchanged.
