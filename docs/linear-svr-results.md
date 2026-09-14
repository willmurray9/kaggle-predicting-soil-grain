# Linear support-vector regression — 14 September 2026

Nested linear SVR scores **42.96239 local EMD**, slightly better than the
same-input ridge incumbent's **43.40571**. The gain is only **0.44331 EMD**, ten
soils improve, and camera disagreement rises from **29.11306 to 39.24129**.
Without its largest beneficiary, the average gain becomes negative. **All four
submission criteria fail, so no candidate is uploaded.** RGB + texture remains
best publicly at **61.11357**, with one of today's five slots used.

The [declaration](roadmap.md#linear-svr-on-rgb--texture--declared-2026-09-14) was
committed as `1339af5` before scoring. Producing code is
`f56401e77b50e7ed57861d67dc44795efe413a3d`. The C grid, tolerance, solver settings,
and submission criteria remained fixed after the results.

## Controlled comparison

Reuse the exact 17 float64 RGB + physical-texture features, 100 mm center crops,
and equal photo averaging. Reconstruct the alpha-10 ridge incumbent's saved
OOF, camera, and test predictions before fitting SVR; maximum difference is
**1.14e-13**. Every photo of a held-out soil remains excluded from training.

Fit ten separate `sklearn.svm.SVR(kernel="linear")` models to the first ten raw
percentage targets. Use one shared C selected from **0.1, 1, 10**, fixed
**epsilon = 1 percentage point**, stopping tolerance **1e-6**, at most **1,000,000
iterations**, and shrinking enabled. The intercept is unpenalized. Each training
fit learns its own feature mean/std; there is no target scaling or PCA. Keep
clipping, monotone accumulation, and the final 100% endpoint unchanged.

The epsilon-insensitive loss ignores errors within one point and grows linearly
beyond that tolerance. This is closer to the competition's absolute-error
penalty than squared loss, but is not the same objective. Smaller C means
stronger regularization. See the official [SVR API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
and [loss formulation](https://scikit-learn.org/stable/modules/svm.html#svr).

Choose C using inner whole-soil LOO EMD on each outer fold's other 23 soils,
refitting scaling and all regressions on the 22 inner training soils. Exact
ties prefer smaller C. Refit the chosen C on the 23 outer training soils and
use those same models for pooled and separate-camera predictions. Select C
again on all 24 training soils for the final test fit. Every solver fit must
converge; warnings or unsuccessful status abort the experiment. All fits completed.

## Results and submission screen

| Model on the same 17 inputs | Outer EMD, 24 soils | Camera disagreement, 21 pairs |
| --- | ---: | ---: |
| Ridge incumbent, alpha 10 | 43.40571 | **29.11306** |
| Nested linear SVR | **42.96239** | 39.24129 |

| Criterion versus ridge incumbent | SVR result | Decision |
| --- | ---: | --- |
| Mean gain ≥ 1 EMD | +0.44331 | Fail |
| At least 12/24 soils improve | 10 improve, 14 worsen | Fail |
| Camera disagreement ≤ 29.11306 | 39.24129 | Fail |
| Positive gain without largest beneficiary | −1.16793 | Fail |

H666 improves by **37.50193 EMD**, exceeding the entire net gain summed over all
24 soils. Other gains include H516 (+13.52132) and H038 (+11.94610). H371 worsens
by 9.93013, H615 by 9.53698, and H374 by 7.51367. Across the other 23 soils after
omitting H666 from a sensitivity calculation, SVR is **1.16793 EMD worse** on
average. H666 remains in fitting and primary evaluation.

Camera disagreement worsens most for H371 (+63.64 EMD) and H372 (+41.89).
Together they account for about half the net increase across the 21 paired
soils; the reported camera score continues to include every pair.

The small average improvement therefore does not represent a broadly better
model. The closer training loss alone did not improve the majority of soils
or camera stability. These results do not establish whether the main limit is
the descriptors, regularization, sample coverage, or another model assumption;
they provide insufficient evidence for a selective upload.

## Regularization selection

C **1 and 10 each win 12 outer folds**. C 0.1 wins none. Final selection chooses
C **10**, with only a 0.19955 EMD advantage over C 1 in the full-training selection
scores below. These choose the final model; they are **not outer validation
estimates**:

| C | Full-training selection EMD |
| ---: | ---: |
| 0.1 | 62.13413 |
| 1 | 41.54959 |
| 10 | **41.35004** |

The even split of outer choices and small final selection gap do not establish
a uniquely best regularization strength. No additional C values or epsilon
settings were evaluated after these scores.

## Reproduction and verification

Update the environment with `uv sync --extra dev --extra vision`, preserving
our existing encoder setup, then run `make linear-svr`. The command uses the
combined cache from `make texture-kernel` and writes separate OOF/camera curves,
all selections, per-soil comparisons, candidate, and decision under
`artifacts/experiments/linear_svr/`. It does not upload to Kaggle.

- All **199 tests pass**, with both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34897082724) passing. Coverage includes analytical two-soil solutions, intercept
  translation, constant targets/features, query independence, solver failures,
  nested isolation, source alignment, reference reconstruction, and test-data
  independence. Independent implementation review found no actionable issues;
  a separate output audit reproduced the metrics, selections, and no-upload decision.
- All **130 prior artifacts remain byte-identical**. The manifest fingerprints
  11 inputs and the valid ten-row candidate, and records the numerical runtime.
- Added scikit-learn and required runtime dependencies. Every earlier locked
  package version was retained, including NumPy and the vision packages.
  This run used Python 3.11.13, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, and
  scikit-learn 1.9.1; the updated lockfile records supported-platform resolutions.

Unsubmitted file: `artifacts/submissions/rgb_texture_linear_svr.csv`.
SHA-256: `b84d6d42b17173479e97a0d4dfcaf484fd7dacbba716737dd1144826b758a6ac`.

Next: a small shallow-tree ensemble comparison on these compact inputs, with
its settings and budget declared before scoring. That tests another regression
family while keeping image information fixed. The official-preprocessing encoder
candidates remain preserved; neither those nor this SVR candidate is promoted
by this batch.
