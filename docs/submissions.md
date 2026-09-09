# Kaggle submission log

Competition: [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).
Results can be inspected on the authenticated [submissions page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/submissions).

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
