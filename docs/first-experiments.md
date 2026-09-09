# Camera, grayscale, and ridge experiments — 2026-09-09

Completed the six comparisons specified in the [roadmap](roadmap.md). The original
45.24 EMD baseline remains the reference. The new results reveal tradeoffs between
local accuracy and camera sensitivity; they do not establish a Kaggle winner.

## Results

All scores use leave-one-physical-soil-out validation on the same 24 soils.
Each fold excludes every photo and camera view of that soil from training.
Settings were fixed before scoring; ridge uses alpha 10 without tuning.

| Experiment | Held-out EMD ↓ | Camera disagreement EMD ↓ | Test soils with H374 among neighbors |
| --- | ---: | ---: | ---: |
| Original RGB, 100 mm, 3-NN | 45.24 | 33.61 | 9/10 |
| Ridge on original RGB features | 41.21 | 28.89 | N/A |
| Grayscale, 100 mm, 3-NN | 42.98 | 33.08 | 9/10 |
| Normalized grayscale, 100 mm, 3-NN | 48.34 | 25.36 | 2/10 |
| Grayscale, 50 mm, 3-NN | **39.92** | 43.48 | 10/10 |
| Grayscale, 150 mm, 3-NN | 41.60 | 32.67 | 3/10 |

Camera disagreement is the mean EMD between the two predictions obtained from
Motorola and Samsung views of each of the 21 soils photographed with both phones.
For each soil, both predictions use the same other 23 soils as training data.
Lower disagreement means more consistent predictions, not necessarily more
accurate predictions: a constant predictor would have zero disagreement.
Views also differ in scene framing and photo selection, so this diagnostic does
not isolate camera hardware alone.

## What changed our understanding

**The smallest crop wins on the local mean, but has the worst camera consistency.**
Its 39.92 EMD is 11.76% below the reference. It improves 10 soils, worsens 8, and
ties on 6. H374 appears in all ten test neighborhoods, and camera disagreement
rises from 33.61 to 43.48. Keep it as a candidate rather than automatically
promoting it over the reference.

**Ridge is a useful comparison, but its overall gain is concentrated in H374.**
The 41.21 EMD is an 8.92% reduction. Ridge improves 13 soils and worsens 11;
H374's error drops from 126.53 to 26.72. Averaging only the other 23 held-out
predictions gives 41.71 for the reference and 41.84 for ridge. This is a diagnostic
slice of the existing predictions, not retraining after removing H374. H374 stays
in every reported full-set score.

**Removing color alone makes a modest difference.** The 100 mm grayscale model
scores 42.98 but retains H374 in 9/10 test neighborhoods. The 150 mm crop reduces
that concentration to 3/10 and scores 41.60, with camera disagreement similar to
the reference. Its improvement is also strongly influenced by H374.

**Contrast normalization trades accuracy for consistency.** Per-photo grayscale
normalization gives the lowest camera disagreement, but worsens local EMD to
48.34. Brightness/contrast appears to contain useful signal as well as potential
camera sensitivity; this experiment does not establish that it should be removed.

## Decision and next work

Preserve the original baseline and retain the new candidates for comparison.
The training camera check remains a proxy: test photos use iPhones, absent from
training. Six-way exploratory comparison also makes the minimum score optimistic
as an estimate of a selected model's future performance.

Before increasing model complexity, inspect the worst repeated failures (notably
H038 and F827), paired-camera crops, and H374's influence. A follow-up could combine
information from multiple physical crop sizes, but it should be declared as a new
experiment rather than tuned against these six scores. Kernel ridge and pretrained
features remain later stages of the roadmap.

## Reproduce and inspect

```bash
make experiments
make test
```

The command uses the prepared tables and photo index from `make data`. It extracts
each feature configuration once, writes per-photo features, evaluates the six
fixed models, and validates separately named candidate submissions. There is no
new dependency, hyperparameter search, or automatic model promotion.

- [Summary table](../artifacts/experiments/first_batch/summary.csv)
- [Per-soil held-out predictions and errors](../artifacts/experiments/first_batch/oof_predictions.csv)
- [Held-out camera predictions](../artifacts/experiments/first_batch/camera_predictions.csv)
- [Camera-specific errors and sample counts](../artifacts/experiments/first_batch/camera_summary.csv)
- [Test nearest neighbors](../artifacts/experiments/first_batch/test_neighbors.csv)
- [Run manifest](../artifacts/experiments/first_batch/manifest.json)
- [Ridge submission](../artifacts/submissions/experiments/ridge_rgb_100.csv)
- [50 mm grayscale submission](../artifacts/submissions/experiments/gray_50.csv)

Verification: all **31 tests pass**; all six candidate submissions and held-out
curve tables are valid; code review found no blocking issues. Rerunning the
original model reproduces **45.24084370102147 EMD**, and its five saved reference
artifacts remain byte-for-byte identical. The competition archive fingerprint
is recorded in [the baseline report](baseline-results.md).
