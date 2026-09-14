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

## Experiments

The [experiment review](docs/experiment-review.md) explains all completed results,
the likely limits of the current models, and the remaining experiments.
The [roadmap](docs/roadmap.md) records the model progression and validation rules.
Run the fixed camera/grayscale/ridge comparisons with:

```bash
make experiments
```

See [first-batch results](docs/first-experiments.md) for accuracy and camera
sensitivity. Detailed outputs are in `artifacts/experiments/first_batch/`, with
candidate submissions in `artifacts/submissions/experiments/`. The original
baseline outputs are preserved.

Run the fixed grayscale crop blend and visual error audit after that batch:

```bash
make multicrop
make audit
```

The [multi-crop results](docs/multicrop-results.md) record the submission decision;
the [error audit](docs/error-audit.md) documents the difficult soils. The blend
scores 40.12 local EMD versus the original reference's 45.24.
Its first Kaggle submission scored 76.75797 public EMD; the
[submission log](docs/submissions.md) records the file hash, code commit, and result.

`make camera-balance` runs the next controlled comparison using those saved
features. [Equal camera weighting](docs/camera-balance-results.md) worsened both
local measures, so the existing submission remains our reference.

`make kernel-ridge` compares one fixed RBF kernel model on the original RGB
features. [Kernel-ridge results](docs/kernel-ridge-results.md) show worse local
accuracy despite improved camera agreement; this candidate was not submitted.

For the optional frozen ResNet-18 baseline:

```bash
uv sync --extra dev --extra vision
make frozen-model
```

The first run downloads the official pretrained checkpoint. See
[frozen-feature results](docs/frozen-resnet-results.md) for the checked competition
rules, reproducibility details, and the decision to retain the current submission.

`make nested-ridge` reuses that feature cache to select regularization inside
each training fold. It needs no vision packages once the cache exists. The
[nested comparison](docs/nested-ridge-results.md) improved camera agreement but
not accuracy, so it also retains the current submission.

`make model-blend` evaluates the fixed 50/50 blend of multi-crop and nested
ridge predictions. `make nested-neighbors` chooses one shared neighbor count
inside each outer training fold. [Both comparisons are complete](docs/blend-and-neighbors-results.md):
the blend worsened accuracy, while neighbor tuning selected the existing count
of three in every fold.

`make spatial-coverage` extracts five fixed positions per photo at each existing
crop size and averages their grayscale features. It retains three neighbors and
whole-soil validation. Outputs live in `artifacts/experiments/spatial_coverage/`
and `artifacts/submissions/spatial_multicrop.csv`.
The [spatial results](docs/spatial-coverage-results.md) show 42.53062 local EMD
and 25.20834 camera disagreement.

The [latest submission comparisons](docs/submissions.md) test candidates even
when their local scores are worse: RGB ridge reached **61.87967 public EMD**
and nested ResNet ridge **63.01764**, both improving on the original 76.75797.
Spatial coverage scored **82.91954**, leaving RGB ridge best at the end of that batch.
The [roadmap](docs/roadmap.md) records the batch before those uploads.

`make physical-texture` runs the fixed RGB/grayscale comparison with and without
four contrast-normalized texture measurements at physical offsets. It reuses
the first-batch feature caches, retains ridge alpha 10, and verifies the original
RGB predictions before producing three new candidates. **RGB + texture is now
our best public submission at 61.11357 EMD**, a 1.24% improvement over RGB ridge.
Grayscale and grayscale + texture scored 70.83530 and 71.34901. See the
[texture results](docs/physical-texture-results.md) for the controlled comparisons
and `artifacts/experiments/physical_texture/` for detailed outputs.

`make pca-ridge` tests eight-component PCA fitted inside every training fold,
with the existing nested ridge penalty selection, plus a fixed 50/50 blend with
RGB + texture. It reconstructs the uncompressed reference and applies a
predeclared screen before recommending at most one submission. The
[PCA report](docs/pca-ridge-results.md) records 45.02758 local EMD for PCA and
42.02843 for the blend; only the blend passes all four submission criteria.
