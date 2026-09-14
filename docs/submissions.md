# Kaggle submission log

Competition: [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).
Results can be inspected on the authenticated [submissions page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/submissions).

## 2026-09-14 — selective PCA blend

The user requested selective submissions with evidence of a possible improvement.
The [PCA comparison](pca-ridge-results.md) and four-part screen were declared in
`93de6fef6a0b7f99ded6432b3f69ec383bc1a64c` before new scores, allowing at most
one upload. The standalone PCA model failed that screen. Its fixed 50/50 blend
with RGB + texture passed: local EMD 42.02843 versus 43.40571, 14/24 soils better,
camera disagreement 17.90374 versus 29.11306, and +0.47653 mean improvement after
omitting the largest beneficiary from the comparison. Every soil remains in
model fitting and the primary validation score.

The selected candidate is `artifacts/submissions/rgb_texture_pca8_blend.csv`,
produced by `8b47558d6910ef89d391c9f72b3a964894eb02c4`, with SHA-256
`52c41bce5cbd43a2d31c0a831f6a881cab5666061a94f7e3c9964d79aba44549`.
All 136 tests and both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34886175828)
passed before upload. Independent review reproduced the decision from saved
curves; the file validates and changes every test curve relative to all seven
earlier submissions.

| Field | Result |
| --- | --- |
| Submission ID | `56237810` |
| Submitted UTC | `2026-09-14 19:21:46.313000` |
| Status | Complete |
| Public EMD | **63.33017** |
| Prior best public EMD | **61.11357**, RGB + texture |
| Private score | Unavailable |
| Local outer EMD | 42.0284318338925 |
| Camera disagreement | 17.90373798556798 |

The blend worsens public EMD by **2.21660 (3.63%)** despite meeting the local
screen. Retain RGB + texture as our best submitted model. The local gains were
real for these folds, but they did not transfer to this public population. One
result cannot determine whether the difference arises from cameras, soil
composition, or exploratory selection. Do not change blend weights, PCA counts,
or the screen in response to this result.

Final API receipt:
`artifacts/experiments/informative_submissions/2026-09-14_rgb_texture_pca8_blend.json`.
It records the input checks, screen evidence, producing/declaration commits,
candidate hash, request/response timestamps, and final server status and score.

Preflight confirmed seven lifetime uploads, none today, and a daily limit of
five. After this upload there are eight lifetime submissions, one today, and
four remaining daily slots. No additional submissions or post-result
weight/component searches are part of this batch.

## 2026-09-11 — physical texture and color control

The [four-configuration comparison](physical-texture-results.md) fixes ridge
alpha 10 and the original center 100 mm crop. It separates removing color from
adding four contrast-normalized physical texture measurements. The three new
uploads were declared in `34a0bfb83340f5504fcf48a219d011b7565b3e3c` before any
new validation or public score, with no requirement to improve local metrics.

Producing code: `b3da3cf6533c4faf7b451c5e4be827c0556e157d`. All 109 local tests
and both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34609779482)
passed before upload. The RGB reference reconstructed OOF, camera, and test
predictions within 5.69e-14. All 172 input fingerprints matched; each candidate
passed schema and curve checks and differed from every prior submission and
each other at absolute tolerance 1e-9.

| Model | Local EMD | Camera disagreement | Public EMD | Submission ID |
| --- | ---: | ---: | ---: | --- |
| RGB ridge reference (September 10) | 41.20591 | 28.89103 | 61.87967 | `56152621` |
| RGB + physical texture | 43.40571 | 29.11306 | **61.11357** | `56167133` |
| Grayscale ridge | 43.15793 | 22.74678 | 70.83530 | `56167136` |
| Grayscale + physical texture | 45.34803 | 22.32622 | 71.34901 | `56167137` |

All three submissions are complete; private scores remain unavailable. RGB +
texture is the new best public submission, improving by **0.76610 EMD (1.24%)**
over RGB ridge despite its worse local score. Adding texture to grayscale instead
worsens public EMD by 0.51371. Removing color worsens both local and public
accuracy in this fixed comparison, despite better camera agreement. The small
public gain does not establish a private-test improvement or a universal value
for these descriptors. No settings were selected using intermediate scores.

| Candidate file in `artifacts/submissions/` | Uploaded UTC | SHA-256 |
| --- | --- | --- |
| `ridge_rgb_texture_100.csv` | `2026-09-11 14:24:55.030000` | `f1a63c3591ac93d938f5ed64947be8cd481e62895c71febd5e365a004675911b` |
| `ridge_gray_100.csv` | `2026-09-11 14:25:00.667000` | `dc0bead104b134795a2c5c6011a283c4ae241c44fa38ed3d1e7afd59963c5d63` |
| `ridge_gray_texture_100.csv` | `2026-09-11 14:25:02.827000` | `f6e61138d27731d2641157c9e6dde81a85c89cc946eb7e4494f9be2faa522fad` |

Final API receipts are in `artifacts/experiments/informative_submissions/`, named
`2026-09-11_ridge_rgb_texture_100.json`, `2026-09-11_ridge_gray_100.json`, and
`2026-09-11_ridge_gray_texture_100.json`. Each records the producing/declaration
commits, exact file hash, local metrics, request time, API reference, and final
server timestamp, status, and scores.

Authenticated preflight confirmed zero uploads today and a daily API limit of
five. The completed batch used the conservative budget of three: seven lifetime
uploads, three today. Candidate choices and settings remained unchanged regardless
of intermediate public outcomes. No further uploads in this batch.

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
