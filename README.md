# Soil Grain Size Kaggle Scaffold

Learning-first scaffold for the Kaggle competition [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).

The first phase stops before training a real computer vision model. It gives you:

- clean data and artifact folders,
- local target and submission validation,
- the competition metric,
- image-blind baselines,
- visual EDA reports for deciding whether to keep competing.

## Setup

```bash
/Users/wmurray/.local/bin/uv sync --extra dev
```

## Data

Either configure Kaggle credentials at `~/.kaggle/kaggle.json` and run:

```bash
make download
```

or manually place the Kaggle files in:

```text
data/raw/latest/
```

Expected raw files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `ppm.csv`
- `All_Photos_training/`
- `All_Photos_test/`

## Workflow

```bash
make data
make labels
make baselines
make eda
make test
```

Generated outputs live in `artifacts/`, especially:

- `artifacts/reports/photo_index.csv`
- `artifacts/reports/label_curves.png`
- `artifacts/reports/bin_mass_heatmap.png`
- `artifacts/reports/photo_contact_sheet.png`
- `artifacts/reports/go_no_go.md`
- `artifacts/submissions/`

Validate any submission with:

```bash
make validate SUBMISSION=artifacts/submissions/mean_baseline.csv
```
