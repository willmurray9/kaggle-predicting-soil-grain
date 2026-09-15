# Spectral texture with weighted photo training — 15 September 2026

The fixed combination improves local EMD from **40.53927 to 38.85735** and
camera disagreement from **25.33273 to 13.50764**. It improves 15 of 24 soils
and passes the declared two-metric submission rule plus all four earlier
diagnostic screens. The one selected upload scores **59.84769 public EMD**,
worse than spectral ridge's **55.78511**. Retain spectral ridge as our best
submitted model; no further upload or setting change follows this result.

Declaration: `d49b7d3fdac58bae1dac7b175887cc8922466d03`.
Producing code: `26923069cb63ce88e39d4081baa1f362e094c75f`.
See [the fixed plan](roadmap.md#spectral-features-with-photo-training--declared-2026-09-15)
and [the preceding two experiments](physical-photo-results.md).

## What changed

Combine the existing 23 RGB/texture/spectral features with the existing
weighted-photo ridge. Each training soil contributes total weight one across
its photos, scaling is fitted to training-soil feature means, and alpha remains
10 with an unpenalized intercept. Average query features before curve repair.
The six spectral bands, 100 mm crop, labels, and all other settings stay fixed.
There is no blend or parameter search, and no new features or model weights
are extracted or downloaded.

This asks whether broader texture information remains useful while penalizing
predictions that change between views of the same soil. Such variation includes
real grain arrangement and heterogeneity as well as cameras and framing; the
penalty can discard useful information. All photos of each held-out soil remain
excluded. There are still only 24 independent labels, and the camera check does
not directly test transfer to the iPhones in the competition test data.

The incumbent for every primary comparison is the **23-feature spectral ridge**
that scored **55.78511 publicly**, not the older 17-feature model. Fresh preflight
at 16:27–16:28 UTC found us at **83/249**, with one of five daily slots used.
All 165 published data paths/sizes and all seven official pages were unchanged.

## Local results

Lower is better. Both rows use the same 24 held-out soils and 21 camera pairs.

| Model | Held-out EMD | Camera disagreement |
| --- | ---: | ---: |
| Spectral ridge reference | 40.53927 | 25.33273 |
| Spectral ridge trained on weighted photos | **38.85735** | **13.50764** |

| Diagnostic | Result |
| --- | ---: |
| Mean EMD improvement | +1.68193 |
| Soils improved | 15/24 |
| Mean improvement excluding largest beneficiary | +0.73119 |
| Earlier four-screen diagnostic | Pass |
| Declared strict improvement in both metrics | Pass |

H666 contributes the largest gain, **+23.54885 EMD**. H366 improves by
**8.15870** and H038 by **6.93323**. The largest losses are H668 (**−8.85642**),
H549 (**−6.88475**), and F827 (**−4.62932**). The gain remains positive without
H666, but it is smaller. All soils remain in fitting and the primary score.

The two ideas are compatible in this local comparison. The combined model also
has lower mean EMD and camera disagreement than the earlier 17-feature photo
model (42.28823 / 13.91782); that is a secondary descriptive comparison, not a
separate selection rule. These exploratory comparisons do not make the local
score an unbiased estimate of future performance.

## Kaggle result

Submission **56259049**, uploaded **2026-09-15 at 16:35:22.640 UTC**, completed
at **59.84769 public EMD**. That is **4.06258 EMD (7.28%) worse** than the
spectral incumbent. The private score is unavailable. See the
[submission log](submissions.md) and the [authenticated submissions page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/submissions).

The local gain did not transfer to the public evaluation. Stronger agreement
between training-camera views was insufficient evidence of transfer to this
test population. The photo penalty may suppress useful signal, or the local
and public populations may differ; this result does not distinguish those
explanations. Keep the spectral model and its settings unchanged. No penalty,
weight, or feature search was run in response to this outcome.

Kaggle now shows **eleven lifetime submissions, two today, and three daily
slots remaining** under the API limit of five. The finalized receipt is
`artifacts/experiments/informative_submissions/2026-09-15_spectral_photo_ridge.json`.
It was written before the request and updated against the exact accepted
reference, recording hashes, code/declaration commits, selection evidence,
CI, timestamps, and the completed score.

## Reproduction and verification

Run `make spectral-photo` after `make physical-photo`. The command verifies the
previous manifest, independently aligns both feature caches by split, soil,
camera and path, and reconstructs spectral OOF/camera/test curves within
**5.69e-14** before evaluating the combination. It writes separate OOF, camera,
per-soil comparison, candidate and summary files. It never uploads automatically.

All **299 tests pass**, including 11 focused new cases for cache alignment,
source/reference corruption, preserved inputs, and independence of held-out
predictions from test features. Independent implementation review found no
issues. Both [CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34995667798)
passed before upload. Independent output review confirmed the metrics,
24 unique OOF rows, 45 camera rows, 21 pairs, exact submission template order,
and predictions numerically distinct from all ten earlier uploads. All **159
prior artifacts remain byte-identical**; all **176 recorded source/photo/candidate
hashes** match. No existing predictor or dependencies changed.

Candidate: `artifacts/submissions/rgb_texture_spectral_photo_ridge.csv`.
SHA-256: `456dcda4ef020885cc97144c5dbe59e1dbcc02fb8055657475224cfd80a8bfeb`.
Detailed artifacts: `artifacts/experiments/spectral_photo/`.

The next larger representation option remains self-supervised DINO patch
features with fold-fitted dimensionality reduction. No DINO model or further
spectral/photo-training variation was evaluated in this batch.
