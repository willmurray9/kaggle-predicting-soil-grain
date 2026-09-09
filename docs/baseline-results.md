# First real-data run — 2026-09-09

The first image baseline achieves **45.24 EMD**, compared with **89.89** for the
training-mean baseline, under leave-one-physical-sample-out validation. Lower is
better. This is a useful starting point for image modeling, with significant
uncertainty about transfer to the test cameras.

## Data audit

- Current archive: 24 labeled samples / 127 photos; 10 test samples / 35 photos.
- All 162 photos are readable and match a sample and camera. Every sample has
  photos; there are no exact duplicate image files (SHA-256 comparison).
- Training cameras: Motorola Edge (69 photos), Samsung A52 (55), and Motorola
  Edge 60 Fusion (3, all of sample H374). Test cameras: iPhone 14 (14) and
  iPhone 16 (21).
- Training photos are mostly downsampled. The image baseline corrects native
  camera PPM by the actual/native long-edge ratio before cropping. This assumes
  the provided images were uniformly resized, without an additional crop.
- Updated filenames and the absence of a separate test CSV are handled in
  `configs/data.yaml`. Test IDs are taken directly from the submission template.
- The Münster photo names match the Muenster submission ID after normalization;
  phone names, rather than internal camera-model names, identify PPM rows.

Archive SHA-256:
`e041a22192b943e39d0c6486e6a6879aa12901089df7af2c28c4f8bfa9403b68`

## Validation results

| Method | EMD | Evaluation |
| --- | ---: | --- |
| Equal mass per bin | 102.02 | Fixed prediction on training samples |
| Training mean | 89.89 | Leave one sample out |
| Training median | 97.25 | Leave one sample out |
| Image features + 3 nearest samples | **45.24** | Leave one sample out |

The image baseline reduces mean EMD by **49.67%** and beats the mean baseline on
**20 of 24** held-out samples. These are local validation scores, not Kaggle scores.

Each photo is cropped to a central 100 mm square and resized to 256 × 256 pixels.
Its 13 features are RGB 10th/50th/90th percentiles, grayscale standard deviation,
and mean grayscale differences at pixel offsets 1, 4, and 16. Photo features are
averaged per sample. Each fold fits feature scaling on the other 23 samples and
averages the target curves of the three nearest samples. The held-out sample's
photos never enter training. These settings were fixed before evaluating; no
hyperparameter search or pretrained model was used.

The largest errors are H374 (126.53), H038 (119.33), and F827 (107.40).
H374 is also the only training sample from its camera, making it a useful case
for investigating camera/color sensitivity. It remains in the reported score.
It appears among the three neighbors for 9 of the 10 test samples, reinforcing
the need to investigate that sensitivity before interpreting test predictions.

## Interpretation

The photos visibly separate several fine and coarse soil regimes, and the model
captures useful signal. However, 24 samples provide a small validation set, and
the test cameras are absent from training. Holding out soil samples does not
measure transfer to those new cameras or rule out similarity between specimens.
Exact-file deduplication does not detect all near-duplicate scenes.

The next experiment should investigate color/camera sensitivity and the largest
residuals before adding model complexity. Preserve this fixed baseline as the
comparison point.

## Reproduce and inspect

```bash
make download
make data labels baselines eda image-model
make validate SUBMISSION=artifacts/submissions/image_baseline.csv
make test
```

- [Data coverage](../artifacts/reports/data_summary.json)
- [Training contact sheet](../artifacts/reports/photo_contact_sheet.png)
- [Test contact sheet](../artifacts/reports/test_photo_contact_sheet.png)
- [Label heatmap](../artifacts/reports/bin_mass_heatmap.png)
- [Model summary](../artifacts/reports/image_model_summary.json)
- [Per-sample validation errors](../artifacts/reports/image_validation.csv)
- [Held-out predictions](../artifacts/reports/image_oof.csv)
- [Validated image submission](../artifacts/submissions/image_baseline.csv)

The full workflow and all 24 tests pass in the existing Python 3.11 environment.
Generated data and reports are ignored by Git; rerun the commands to recreate them.
