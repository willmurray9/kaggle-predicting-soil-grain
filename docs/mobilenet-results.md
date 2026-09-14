# Second frozen encoder — 14 September 2026

Frozen MobileNetV3-Large with PCA/ridge scores **41.82774 local EMD** and
**13.08667 camera disagreement**. It improves accuracy over RGB + texture ridge
(43.40571), but the saved official ResNet/RGB-texture blend has lower local error
(40.25638). The rule declared before this run therefore selects that saved blend
for the user's requested informative submission.

That upload completed as **56239628**, scoring **62.65462 public EMD**. It is
better than the earlier PCA blend's 63.33017, but worse than the RGB + texture
incumbent's **61.11357**. Today's total is **two of five**, with no further
submissions in this batch.

Declaration: `0ac12929405c3a0f75e2b93838124fafed785932`.
Producing code: `20bd9ba7058ccfeb5859efd74f0e3c6305e1fc89`.
See the [declared experiment and upload rule](roadmap.md#second-frozen-encoder-and-one-informative-upload--declared-2026-09-14).

## Controlled comparison

Use torchvision's exact `MobileNet_V3_Large_Weights.IMAGENET1K_V1` weights,
remove the entire classifier, and retain **960 pooled features** per photo.
Keep the encoder frozen in CPU evaluation/inference mode with batches of eight.
The official V1 transform matches the official ResNet recipe: resize the existing
256-pixel calibrated crop to 256, center crop to 224, then ImageNet normalization.
The retained field is nominally 87.5 mm from the original 100 mm crop. See the
[official MobileNet specification](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html).

Keep equal float32 photo means per soil. Fit standardization, eight-component
PCA, and ridge within every training fold. Choose alpha from 10/100/1000 using
inner whole-soil LOO, with exact ties preferring larger alpha. The outer fold
excludes every photo of one soil; its other 23 soils supply training data.
Final penalty selection uses all 24 labeled soils, followed by one test fit.
No whitening, component rescaling, new blend, or fine-tuning is added.

The architecture, pretrained weights, and feature dimension change together.
Comparing against official ResNet PCA keeps the image recipe and regression
procedure fixed; this does not isolate architecture from pretraining.

## Results

Lower is better in both columns. Camera disagreement is a sensitivity diagnostic,
not an accuracy estimate on unseen phones.

| Model | Outer EMD, 24 soils | Camera disagreement, 21 pairs |
| --- | ---: | ---: |
| RGB + texture ridge, public incumbent | 43.40571 | 29.11306 |
| Official ResNet PCA/ridge | 40.80177 | 16.90987 |
| New MobileNet PCA/ridge | 41.82774 | **13.08667** |
| Saved official ResNet/RGB-texture blend | **40.25638** | 19.41314 |

MobileNet improves **14/24 soils** versus RGB + texture. Its mean gain is
**1.57796 EMD**, but falls to **−0.15661** when the largest beneficiary is omitted
from the sensitivity calculation. It passes three of the four old incumbent
screens; that final robustness criterion fails. All soils remain in training
and the primary score.

The strongest gains versus RGB + texture are H126 (+41.47313), H366 (+33.02951),
and H372 (+11.60638). The largest regressions are H615 (−36.22286), H637
(−23.90925), and H405 (−18.73460). Against official ResNet PCA, MobileNet improves
15 soils but worsens mean EMD by **1.02597**; large losses on H615 and H637
outweigh several smaller gains. Its better camera agreement therefore does not
translate into better average accuracy on this comparison.

Alpha **1000 wins 21 outer folds**, and **100 wins three**. Final selection
chooses **1000**. The following values choose the final penalty; they are not
additional outer validation estimates:

| Alpha | Full-training selection EMD |
| ---: | ---: |
| 10 | 43.24698 |
| 100 | 42.54567 |
| 1000 | **41.50561** |

## One informative submission

The latest user request explicitly replaces the earlier conservative upload
veto for this batch. Before scoring MobileNet, we committed to one upload:
choose the lower outer EMD between the new model and the saved official blend,
with a tie preferring the blend. Camera and robustness screens remain recorded
diagnostics. This is an exploratory selection, not an unbiased performance claim.

The selected file is `artifacts/submissions/rgb_texture_official_pca8_blend.csv`,
SHA-256 `ff75b009ae1025bd90264fd95a961ee4b5def5c62d3e2afde20d2f3cbcb2c4fb`.
It was produced by `0cbb5537fdf3dbfca6e0df6b82a582e0dec1f5c4` in the earlier
official-preprocessing batch and is used unchanged. The [submission log](submissions.md)
records the completed upload and leaderboard result.

The upload completed at **2026-09-14 22:09:20.513 UTC**. Public error is
**1.54105 higher** than our incumbent, and **0.67555 lower** than the earlier
legacy-preprocessing blend. The public result therefore supports retaining
RGB + texture as the best submitted model. We do not change preprocessing, blend
weights, or selection settings in response to this result. Private score is
unavailable.

The receipt at
`artifacts/experiments/informative_submissions/2026-09-14_official_recipe_blend.json`
records the request before upload, server acknowledgement, final status/score,
both producing and selection commits, declaration, provenance, and quota.

## Reproduction and verification

Run `make mobilenet-pca` with the existing vision dependencies and official
ResNet artifacts. The first run downloads the official 21.1 MB checkpoint.
Separate features, OOF/camera curves, inner/final penalty scores, comparisons,
the new candidate, and provenance are saved under
`artifacts/experiments/mobilenet_pca/`. The command does not upload automatically.

- All **231 tests pass**, with both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34902302604)
  passing before upload. Tests cover the exact preprocessing, full classifier
  removal, frozen batch-independent inference, source corruption, aligned
  references, test-data exclusion, and informative selection including ties.
- Independent implementation review found no actionable issues; a separate
  saved-output audit reproduced metrics, penalty choices, and upload selection. Previous
  ResNet defaults and artifacts are preserved; no dependencies changed.
- All **142 previous artifacts remain byte-identical**. The new manifest
  fingerprints inputs, all 162 source photos, checkpoint, feature cache, and
  the new candidate. Every generated curve is finite, monotone, within 0–100,
  and ends at exactly 100.

New MobileNet candidate, retained locally:
`artifacts/submissions/mobilenet_v3_large_pca8_nested_ridge.csv`.
SHA-256: `bf0ee7a526cdc412ee735d9c2292609fd5f21437c67e17a42c8591ddf2cb638f`.
Checkpoint SHA-256:
`8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04`.

A later fixed blend of MobileNet and RGB + texture is a possible next controlled
comparison. It has not been computed or submitted; this batch ends after the
one requested upload.
