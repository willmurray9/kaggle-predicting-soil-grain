# Soil Grain Size Kaggle Baselines

Learning-first scaffold for the Kaggle competition [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).

Current best: **55.78511 public EMD**, using RGB + texture + spectral ridge
(15 September 2026). See the [latest results](docs/physical-photo-results.md).

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
The one blend upload scored **63.33017 public EMD**, so RGB + texture retains
the best public result of **61.11357**. Four of today's five slots remain unused.

`make official-preprocessing` applies the official ResNet recipe to the same
calibrated crop, then evaluates the existing PCA/ridge setup and fixed blend.
It needs the vision dependencies and earlier experiment artifacts. The
[recipe comparison](docs/official-preprocessing-results.md) improves local EMD
to **40.80177** for PCA and **40.25638** for the blend. Camera disagreement rises
versus their earlier versions, so both miss the stricter submission screen.
Neither was uploaded; RGB + texture remains the public incumbent.

`make texture-kernel` runs a six-setting nested RBF kernel comparison on the
incumbent's same 17 RGB + texture features, after reconstructing its linear-ridge
predictions. The [kernel report](docs/texture-kernel-results.md) records **48.58500**
local EMD versus the incumbent's **43.40571**, with worse camera disagreement.
All four submission criteria fail; no upload is made. Existing packages suffice.

`make linear-svr` tests three regularization strengths inside whole-soil folds,
with a fixed one-percentage-point error tolerance and checked solver convergence.
It reuses the 17-feature cache and adds scikit-learn through the lockfile. The
[SVR report](docs/linear-svr-results.md) records **42.96239** local EMD, but only
ten soils improve and camera disagreement increases. All four submission criteria
fail, so no upload is made. Refresh dependencies before running the new command.

`make shallow-trees` evaluates one fixed ensemble on those same 17 inputs:
256 trees, depth at most three, and at least three soils per leaf. The
[tree report](docs/shallow-trees-results.md) records **47.52880** local EMD and
**25.45697** camera disagreement. Accuracy worsens and only 11/24 soils improve,
so no upload is made. It reuses installed packages and preserves prior artifacts.

`make mobilenet-pca` compares a second frozen encoder using the same official
image recipe and nested PCA/ridge head. [MobileNet results](docs/mobilenet-results.md)
are **41.82774** local EMD and **13.08667** camera disagreement. Under the user's
new request for one informative upload, the saved official ResNet blend wins the
declared comparison and scores **62.65462 publicly**. RGB + texture remains best
at **61.11357**; two of five slots were used on September 14.

`make physical-photo` tests six calibrated spectral texture bands and, separately,
ridge trained on individual photos with equal total weight per soil. Both improve
local EMD and camera disagreement. The [comparison](docs/physical-photo-results.md)
records **40.53927 / 25.33273** for spectral ridge and **42.28823 / 13.91782** for
photo training. The selected spectral model scores **55.78511 publicly**, an
**8.72% improvement** over the previous best. One of five daily slots was used
on September 15. No dependencies or pretrained weights were added.

`make spectral-photo` tests their fixed combination: the same 23 spectral inputs
with weighted-photo ridge. The [combination report](docs/spectral-photo-results.md)
records **38.85735 local EMD / 13.50764 camera disagreement**, both better than
spectral ridge. The selected upload scores **59.84769 publicly**, so spectral
ridge remains best at **55.78511**. Two of five daily slots were used on
September 15; no settings changed after the public result.

`make dino-pca` tests frozen self-supervised DINOv2 patch features with the
existing nested PCA/ridge head. The [DINO report](docs/dino-results.md) records
**41.66782 local EMD / 22.10168 camera disagreement**. Mean EMD is worse than
spectral ridge, so the declared rule rejects an upload; only 8/24 soils improve. The candidate
and pinned code/weight provenance are saved. Spectral ridge remains best at
**55.78511 publicly**; three of September 15's five submission slots remain.

`make geometry-transport` tests two different directions: twelve standalone
image-granulometry measurements, and ridge predicting log-size quantiles instead
of CDF heights. The [comparison](docs/geometry-transport-results.md) scores
**48.33618 / 47.47930 local EMD**, with worse camera disagreement for both.
Neither qualifies for upload. The current public best remains **55.78511**;
three daily slots are preserved. There is no parameter or blend search.

`make patch-mixture` adds six measures of within-photo color/texture variation
to the 23 spectral inputs. The [heterogeneity comparison](docs/patch-mixture-results.md)
scores **41.74919 local EMD / 24.17428 camera disagreement**. Only 9/24 soils
improve; EMD is worse than spectral ridge, so no upload is made. The current
public best remains **55.78511**, with three of September 15's five slots unused.
