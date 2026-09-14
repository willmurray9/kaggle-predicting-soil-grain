# Eight-component PCA and a selective blend — 2026-09-14

Compressing the frozen ResNet features improves their nested local EMD from
**49.91991 to 45.02758**. A fixed 50/50 blend with RGB + texture scores
**42.02843**, versus **43.40571** for our current best public submission's local
predictions. The blend passes all four predeclared submission criteria; PCA alone
does not. Only the blend is selected for a possible upload after verification.

The [declaration](roadmap.md#fixed-pca-and-selective-submissions--declared-2026-09-14)
was committed as `93de6fe` before new scores. It limits this round to **at most
one submission**, with no component-count or weight search after the results.
The [submission log](submissions.md) records external outcomes separately.

## What changed

Reuse the same 512 cached ResNet features, 100 mm center crops, and equal photo
averaging. Standardize each feature using the current training soils, fit SVD
on those standardized rows, and keep the first **eight principal components**.
Project queries using that training basis. Do not whiten or rescale the component
scores. Thus compression removes directions while preserving ridge's penalty
in retained directions. A full-rank projection reproduces ordinary ridge.

Eight is fixed before evaluation. The alpha grid remains 10, 100, and 1000, with
exact ties favoring the larger penalty. Every inner fit relearns its scaler,
PCA, and regression using 22 soils. Each outer fold selects alpha on its other
23 soils, refits there, and predicts the pooled held-out soil plus its camera
views. Every photo of the held-out soil stays excluded. The final test fit uses
all 24 training soils after a separate full-training alpha selection.

The second candidate averages PCA-ridge and the existing RGB + texture curves
with fixed equal weights. OOF soil IDs, camera keys, and test IDs are aligned
before averaging. This is a separate test of complementary errors, with no
fitting or weight search at the blending stage.

## Local results and submission evidence

| Model | Outer EMD | Camera disagreement (21 pairs) |
| --- | ---: | ---: |
| Uncompressed nested ResNet ridge | 49.91991 | 21.19103 |
| Eight-component nested ResNet ridge | 45.02758 | **14.66981** |
| RGB + texture, current public best | 43.40571 | 29.11306 |
| Fixed 50/50 RGB + texture/PCA blend | **42.02843** | 17.90374 |

PCA improves 18 soils and worsens six relative to uncompressed nested ResNet,
reducing mean error by 4.89233 EMD. Against RGB + texture, however, standalone
PCA improves only nine soils and worsens fifteen; its mean error is 1.62188 higher.
It is therefore not eligible for submission in this round.

| Predeclared requirement vs RGB + texture | Blend evidence | Result |
| --- | --- | --- |
| Mean improvement ≥ 1.0 EMD | 1.37727 | Pass |
| At least 12/24 soils improve | 14 improve, 10 worsen | Pass |
| Camera disagreement does not increase | 29.11306 → 17.90374 | Pass |
| Positive mean gain without largest beneficiary | +0.47653 across remaining 23 | Pass |

H126 contributes the largest blend gain, **22.09440 EMD**. Other substantial
gains include H666 (+14.69667) and H183 (+12.09454). G190 worsens by 12.62266,
H516 by 7.32834, and H405 by 7.25085. All soils stay in training and evaluation;
omitting the largest beneficiary is solely a sensitivity diagnostic.

The screen supplies a reason to try one upload, not a significance test or a
guarantee of improving the current **61.11357 public EMD**. The blended local
score also remains above the historical grayscale crop blend's 40.11782, which
scored worse publicly. We use paired evidence without treating local ranking
as an exact predictor of leaderboard ranking.

PCA preserves feature variance, which need not correspond to grain information.
The observed improvement is consistent with discarding unhelpful directions,
but it does not establish what the retained components represent. Complementary
errors let the blend beat either of its two components locally; the public test
will check whether that benefit transfers.

## Alpha selection

Outer folds select alpha **100 for 22 soils** and **10 for two soils**. Final
selection chooses **100**. The full-training selection scores below choose the
final penalty; they are **not** the outer validation estimate:

| Alpha | Full-training selection EMD |
| ---: | ---: |
| 10 | 45.27612 |
| 100 | 44.70085 |
| 1000 | 46.19098 |

## Reproduction and checks

Run `make pca-ridge` after the existing frozen-feature, nested-ridge, and
physical-texture experiments. It needs the saved feature/prediction artifacts
but no vision packages or new image extraction. Outputs live in
`artifacts/experiments/pca_ridge/`: both candidates' OOF/camera predictions,
inner/final alpha scores, per-soil comparisons, and the submission decision.

- The unchanged 512-feature model reconstructs OOF, camera, and test predictions
  within 1.43e-14. All outer/final alpha selections and selection scores match.
- All **136 tests** pass, including PCA scale preservation, full-rank equivalence,
  inner-fold fitting, query/label isolation, invalid component counts, aligned
  blending, reference reconstruction, and every submission-screen criterion.
- Independent implementation review found no actionable issues. All 104 earlier
  artifacts remain byte-identical. The manifest fingerprints 15 inputs and both
  candidate files. Float32 photo aggregation is preserved.
- Both candidates pass submission validation. The selected blend changes all ten
  test curves relative to every one of the seven previously submitted candidates.

| Candidate in `artifacts/submissions/` | SHA-256 |
| --- | --- |
| `frozen_resnet18_pca8_nested_ridge.csv` | `18e30331f3ea6552072a37dc011dc63ae80a10226fca6d9733ba72d96c9a4e45` |
| `rgb_texture_pca8_blend.csv` | `52c41bce5cbd43a2d31c0a831f6a881cab5666061a94f7e3c9964d79aba44549` |

The next separate comparison can test the encoder's official preprocessing
recipe while retaining the same representation and grouped evaluation. Do not
react to this batch by searching new PCA counts or blend weights.
