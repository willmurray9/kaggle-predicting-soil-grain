# Kaggle submission log

Competition: [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).
Results can be inspected on the authenticated [submissions page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/submissions).

## 2026-09-10 — predeclared transfer comparisons

The user authorized informative submissions even when local validation is worse.
The [three-candidate batch](roadmap.md#spatial-coverage-and-informative-submissions--declared-2026-09-10)
was committed as `59620634594795ec8f2037b2d2f3ecb3a0429920` before any new upload.
Both existing candidates reproduced from their feature caches within 7e-14 and
passed schema, ID, ordering, and cumulative-curve checks before submission.

| Model | Local EMD | Camera disagreement | Public EMD | Submission ID |
| --- | ---: | ---: | ---: | --- |
| Original grayscale crop blend | 40.11782 | 31.52595 | 76.75797 | `56131658` |
| RGB ridge, fixed alpha 10 | 41.20591 | 28.89103 | **61.87967** | `56152621` |
| Nested ResNet ridge | 49.91991 | 21.19103 | **63.01764** | `56152632` |
| Five-position grayscale crop blend | 42.53062 | 25.20834 | 82.91954 | `56152697` |

All three new submissions are complete; private scores remain unavailable.
RGB ridge reduces public EMD by 19.4% and nested ResNet ridge by 17.9% relative
to the original submission. Their worse local scores did not imply worse public
performance. This is evidence that our local ranking transfers imperfectly,
without establishing whether cameras, soil composition, or selection effects
cause the difference. The smallest camera disagreement also did not identify
the better of these two public scores. These few public results do not justify
discarding whole-soil validation or tuning against the leaderboard.
The spatial blend worsens public EMD by 8.0% relative to the original despite
improving camera agreement. RGB ridge is our best public submission so far.

RGB ridge provenance:

- Uploaded `2026-09-10 22:34:15.953000 UTC`, file `artifacts/submissions/experiments/ridge_rgb_100.csv`.
- Producing code: `75cd6ae6eb66c1b3e5f2c5ff430968b0c7aa4daf`.
- SHA-256: `afc7edc28793a322038c806b988fa62e8c0683c909dc72dd14d2e8618183608a`.
- API receipt: `artifacts/experiments/informative_submissions/2026-09-10_ridge_rgb_100.json`.

Nested ResNet ridge provenance:

- Uploaded `2026-09-10 22:34:52.737000 UTC`, file `artifacts/submissions/frozen_resnet18_nested_ridge.csv`.
- Producing code: `f7fa0b2fc0b105f3deec7a02361f3b9f49c45272`.
- SHA-256: `cb62d4b0c8e03a8e306e697900cedcd00379ac6751d77d2d0a549e83c1a85838`.
- API receipt: `artifacts/experiments/informative_submissions/2026-09-10_nested_resnet_ridge.json`.

Spatial crop blend provenance:

- Uploaded `2026-09-10 22:39:42.180000 UTC`, file `artifacts/submissions/spatial_multicrop.csv`.
- Producing code: `e7c75db8bdc0f27bb25c5b94b31367f9e0d0a8ea`.
- SHA-256: `9f9b9f20135404110d2fcb6910b3580977dd6549ad80269947ee50d5c36c1e41`.
- API receipt: `artifacts/experiments/informative_submissions/2026-09-10_spatial_multicrop.json`.
- All 98 local tests and both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34538447525) passed before upload.

At the preflight check, Kaggle reported five submissions per day and zero today;
we honored the user's three-upload budget. This batch used exactly three
submissions, for four lifetime uploads. No settings or batch choices changed in
response to intermediate public scores. The original submission and all previous
artifacts remain intact.

## 2026-09-09 — equal-weight grayscale multi-crop

| Field | Value |
| --- | --- |
| Submission ID | `56131658` |
| Submitted at (UTC) | `2026-09-09 22:15:31.063000` |
| Status | Complete |
| Public EMD | **76.75797** |
| Private score | Not available |
| Local leave-one-soil-out EMD | 40.11781628387224 |
| Paired-camera disagreement EMD | 31.52594710152303 (21 pairs) |
| File | `artifacts/submissions/multicrop_baseline.csv` |
| Code commit | `7683ab0819af5d91c25bbe21d54462ce1e0783e5` |
| SHA-256 | `f72fcccbe8371b8d2728823db51117c1c06121b8504f563738b944b311750741` |

The file averages the fixed grayscale 50, 100, and 150 mm 3-NN curves with equal
weights. Its ten rows passed the local submission validator; all 37 tests and
[GitHub CI for the submitted code](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34411205467)
passed before submission. The code commit was pushed using the personal Git
identity. The ignored local API receipt is
`artifacts/experiments/multicrop/kaggle_submission.json`.

The [selection rule](roadmap.md#current-follow-up-error-audit-and-equal-weight-multi-crop-blend)
was committed before computing the blend or seeing any leaderboard result.
The candidate beat the original reference on both declared local criteria; see
[multi-crop results](multicrop-results.md). This is our first submission, so we
have no public-score comparison with our other models.

The public error is higher than local validation. The populations differ, with
unseen iPhone cameras in the test set and only 24 labeled training soils. One
score cannot establish whether camera differences, soil composition, or local
selection effects explain the gap. Treat this as the first external benchmark;
select future candidates through grouped local validation and camera diagnostics.
