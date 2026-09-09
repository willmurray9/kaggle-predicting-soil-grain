# Soil Grain Size Kaggle Baselines

Learning-first scaffold for the Kaggle competition [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).

The pipeline provides:

- clean data and artifact folders,
- local target and submission validation,
- the competition metric,
- image-blind baselines,
- visual EDA reports,
- a small image baseline with validation grouped by soil sample.

## Setup

```bash
/Users/wmurray/.local/bin/uv sync --extra dev
```

## Data

With Kaggle CLI authentication configured, run:

```bash
make download
```

or manually place the Kaggle files in:

```text
data/raw/latest/
```

Expected files in the current competition archive:

- `Training_labels_updated.csv`
- `sample_submission.csv`
- `ppm_updated.csv`
- `Training-All_Photos_updated/`
- `Test_All_Photos/`

Test IDs come from `sample_submission.csv`; no separate test CSV is required.
Paths are configured in `configs/data.yaml`. Data and generated outputs stay out of Git.

## Workflow

```bash
make data
make labels
make baselines
make eda
make image-model
make test
```

Generated outputs live in `artifacts/`, especially:

- `artifacts/reports/photo_index.csv`
- `artifacts/reports/label_curves.png`
- `artifacts/reports/bin_mass_heatmap.png`
- `artifacts/reports/photo_contact_sheet.png`
- `artifacts/reports/test_photo_contact_sheet.png`
- `artifacts/reports/go_no_go.md`
- `artifacts/reports/image_model_summary.json`
- `artifacts/reports/image_validation.csv`
- `artifacts/submissions/`

Validate any submission with:

```bash
make validate SUBMISSION=artifacts/submissions/mean_baseline.csv
```

## Image baseline

Each photo contributes 13 color and texture measurements from a central 100 mm
square resized to 256 pixels. Camera PPM is corrected for the downloaded image's
resolution. Photo features are averaged into one row per physical sample.

The model averages the grain curves of the three nearest training samples.
Feature scaling is fitted within each leave-one-sample-out fold. All photos of the
held-out soil stay out of training; averaging valid curves preserves the submission
constraints. Crop size, features, and neighbor count are fixed, with no tuning.
This baseline uses the existing NumPy/Pillow dependencies and no pretrained weights.

See [the first-run results](docs/baseline-results.md) for scores and limitations.
