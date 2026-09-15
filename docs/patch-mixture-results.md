# Within-photo heterogeneity — 15 September 2026

Adding six measures of regional variation scores **41.74919 local EMD**, worse
than spectral ridge's **40.53927**. Camera disagreement improves slightly, but
only **9/24 soils** improve. The candidate fails the declared submission rule;
**no upload was made**. Our best public EMD remains **55.78511**.

Declaration: `ed431eea003f0ec0c9266e56b3c3eedbbdfb4e1a`.
Producing code: `ffb671a5257d3da6436f3153bfbcfb0416d5d8f7`.
See [the fixed protocol](roadmap.md#within-photo-heterogeneity--declared-2026-09-15).

## What changed

The hypothesis was that averaging image measurements can hide mixtures: a
uniform surface and a photo with distinct fine/coarse regions might have
similar averages. Earlier spatial coverage averaged more crops. This experiment
keeps six measurements of how much local descriptors differ across a photo.

Divide the existing calibrated 100 mm, 256-pixel crop into sixteen fixed
64×64-pixel tiles, each covering nominally 25 mm. For every tile, calculate
three RGB means and three grayscale mean absolute differences at offsets
1, 4 and 16 pixels. Average horizontal and vertical differences, using only
pairs inside the tile. There is no tile resizing or additional normalization.

For each descriptor, take the 75th minus 25th percentile across the sixteen
tiles, with linear interpolation. These six interquartile ranges are computed
within each photo, then photo features are averaged equally per soil. Append
them to the unchanged 23 RGB/texture/spectral inputs, giving **29 features**.
Retain ridge alpha 10, training-only scaling, the unpenalized intercept and
existing curve repair. All 24 soils and 21 camera pairs remain in evaluation;
all views of the held-out soil are excluded from fitting.

These are measurements of surface variation, not grain mass fractions.
Lighting gradients and shadows also create variation. IQR ignores a signal
confined to one to three of sixteen otherwise identical tiles, so it can miss
rare large particles. The descriptor also discards where the differing tiles
occur. These are limits of the fixed representation, not established causes
of its observed result.

## Results

Lower is better. Both rows use the same whole-soil leave-one-out folds.

| Model | Held-out EMD | Camera disagreement | Soils improved versus reference |
| --- | ---: | ---: | ---: |
| Spectral ridge | **40.53927** | 25.33273 | — |
| Spectral ridge + six tile IQRs | 41.74919 | **24.17428** | 9/24 |

Mean improvement is **−1.20991 EMD**. Excluding the largest beneficiary from
the diagnostic gives **−1.49997**; no soils are excluded from actual training
or primary validation. Only the camera criterion passes the four older
diagnostic screens. The authoritative upload rule requires strict improvement
in both mean EMD and camera disagreement, so the candidate is not selected.

The largest gains are H372 (**5.46142 EMD**), H126 (**1.71860**) and F827
(**1.04366**). The largest regressions are H038 (**9.39963 worse**), H516
(**6.06556**) and H374 (**5.25054**). Fifteen soils worsen overall. This
comparison offers no evidence for replacing the current model with the fixed
regional-variation extension. Better camera agreement alone does not establish
better predictions on the unseen iPhone cameras.

No alternative grid, scale, descriptor, spread statistic, regularization or
blend was evaluated after these scores.

The next proposed direction is a **coarse-particle feasibility check** at the
highest available photo resolution. Inspect candidate boundaries on a fixed
small panel of paired-camera crops before fitting any model. If individual
regions correspond plausibly to grains, counts and projected-area/diameter
summaries could retain rare coarse material missed by IQR. This differs from
earlier granulometry, which removed contrast without identifying instances.
Touching grains, shadows and the difference between 2D area and bulk mass are
major risks. Available competition images may already be reduced; no original
camera detail can be recovered. This direction is not yet evaluated, and an
unconvincing boundary audit should stop it before another prediction experiment.

## Reproduction and verification

Run `make patch-mixture` after the preceding experiments. The driver verifies
recorded source/photo hashes, independently aligns the 17-feature and six-band
spectral caches, and reconstructs the reference's held-out, camera and test
curves before new extraction. Maximum difference is **5.69e-14 percentage
point**. Existing model code and package dependencies are unchanged.

Outputs in `artifacts/experiments/patch_mixture/` include the 162×29 photo
feature cache, 24 held-out predictions, 45 camera predictions, per-soil errors
and provenance. The candidate contains ten valid curves in template order.

All **413 tests pass**, including 29 new feature/driver checks. Analytic color
and ramp fixtures verify descriptor values and physical offsets. Other checks
cover equal histograms with different regional organization, sparse outliers,
quarter turns, calibration, exact preservation of all 23 reference features,
source corruption including a refreshed bad-reference hash, and validation
independence from test-photo changes. Independent implementation reviews found
no correctness issues.
Both [CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35009088112)
passed on the producing commit. An independent artifact audit recomputed the
metrics and submission decision, verified all **177 source/photo/feature/candidate
hashes**, and confirmed all **180 earlier artifacts remain byte-identical**.
The original 23 feature columns match their source caches exactly by photo key.

Candidate: `artifacts/submissions/rgb_texture_spectral_patch_iqr_ridge.csv`.
SHA-256: `cf1173e3cbc8e5837137afff437a14978a64644470aea46e19168a57dd2dc024`.
Feature-cache SHA-256:
`bf4a94e3435f0a7163197de4516f1d6c4b2c7ebf3495506f84c06834b1e44ba3`.

The 18:34 UTC Kaggle preflight found eleven completed lifetime submissions,
two of five used today, and three remaining, with rank **83/250**. No upload
was requested by this experiment; fresh history at **18:46:36 UTC** confirms
the same count and best score. Code, tests and this report are tracked in Git;
generated artifacts remain local under the existing repository policy.
