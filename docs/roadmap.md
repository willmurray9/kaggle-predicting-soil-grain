# Competition roadmap

Keep experiments small, reproducible, and grouped by physical soil sample.
There are 24 labeled soils, regardless of how many photos or crops we create.
The fixed starting point is the [first image baseline](baseline-results.md):
**45.24 EMD**, versus **89.89** for the leave-one-sample-out mean. Lower is better.

## Model progression

1. **Camera and feature robustness — first batch complete.** Compare color, grayscale,
   lighting normalization, and physical crop sizes with the same 3-neighbor model.
   Measure whether different cameras produce different predictions for the same
   held-out soil. Investigate H374, which appears in 9/10 test neighborhoods.
2. **Ridge regression — first batch complete.** Learn a regularized linear relationship
   from the original 13 features to the grain curve. Keep image processing fixed
   so the comparison isolates the prediction model.
3. **Kernel ridge — fixed comparison complete.** The RBF candidate scores 52.55
   local EMD and does not replace the submitted blend.
4. **Frozen pretrained image features + ridge — fixed comparison complete.**
   ResNet-18 plus alpha-10 ridge scores 49.65 local EMD; nested regularization
   selection scores 49.92. Neither replaces the submitted blend.
5. **Light fine-tuning — optional.** Adapt a small part of the encoder only if
   frozen features show stable value. Training a large network from scratch is
   not a priority with 24 labeled samples.
6. **Blend complementary models — fixed comparison complete.** The equal blend
   of multi-crop and nested ridge scores 43.74 and does not replace the submission.

The [latest comparison report](blend-and-neighbors-results.md) records the
blend result, nested neighbor selection, and the next spatial-coverage experiment.

## Current follow-up: error audit and equal-weight multi-crop blend

Declared before computing the blend or looking at a public Kaggle score:

- Audit F827, H038, and H374 with actual photos, physical crops, labels, and
  held-out predictions. Treat possible explanations as hypotheses; do not edit
  labels or exclude difficult samples based on poor model fit.
- Average the existing 50, 100, and 150 mm grayscale 3-NN predictions with equal
  one-third weights. Align soil IDs and camera IDs before averaging, keep each
  soil's original held-out fold, and recompute EMD from the averaged curves.
- Record source artifact fingerprints and preserve the existing baseline files.
- Use the multi-crop candidate for one initial Kaggle submission if its local
  EMD and paired-camera disagreement both beat the original reference. Otherwise
  use the existing ridge candidate. The visual audit can reveal a concrete data
  or implementation problem that must be resolved first.
- Record the exact submitted file hash, code commit, submission ID, and public
  result. Treat the public score as an external check, not a new tuning target.

The user has authorized submissions when they make sense and regular remote
updates. Commit and push completed, tested milestones using the personal Git
identity. CI runs on `main` and `codex/**` pushes. No generated data goes into Git.

Kaggle access checked before the first submission on 2026-09-09: account entered;
the competition API lists a 2026-11-30 deadline.

Audit and blend complete: [audit findings](error-audit.md) and
[multi-crop results](multicrop-results.md). The blend scores 40.1178 local EMD
and 31.5259 paired-camera disagreement, beating the original reference on both
predeclared criteria. Its [first Kaggle submission](submissions.md) completed with
public EMD 76.75797. The audit, blend, submission, and provenance record are complete.

## Camera weighting comparison — complete

The audit identified unequal photo counts per camera within a soil. Compare
equal-per-camera feature averaging with the current equal-per-photo averaging:
average each camera's photos first, then average the camera means. Keep the
three grayscale crop models, 3-NN settings, and equal prediction blend fixed.
Use the same 24 held-out soils and 21 paired-camera diagnostics, and report
per-soil changes. This isolates one observed source of representation imbalance
before adding kernel ridge. The public score does not choose settings or weights.

Execution rule recorded on 2026-09-10, before running this comparison: reuse the
saved grayscale photo features and existing grouped evaluator; average features
within `(split, sample_id, camera)` before averaging cameras within a soil. This
applies to both training and test soils. Write separate artifacts, including
per-soil changes and input hashes, while preserving previous outputs. Submit
once only if the fixed blend improves both local EMD (40.11781628387224) and
camera disagreement (31.52594710152303) over the submitted blend, and produces
different test predictions. Otherwise record the result without submitting.

After verification, merge this work into `main`, push, and delete fully merged
development branches. Preserve any branch with unique work and any active
worktree. Use the personal Git identity for all new commits.

Result: local EMD worsened to 42.4228 and camera disagreement to 38.3402. Retain
the photo-weighted blend and make no submission for this candidate. See
[camera-balance results](camera-balance-results.md) for the full comparison.
Next, compare a fixed kernel-ridge model using the same grouped validation;
record its settings before running it.

## Kernel ridge — fixed comparison complete

Declared before computing predictions on 2026-09-10:

- Reuse the original 13 RGB/texture features from the 100 mm crop and equal
  photo averaging. This matches the existing linear ridge inputs.
- Fit one RBF kernel ridge model with `alpha = 1` and `gamma = 1 / 13`, without
  tuning. Standardize features from training soils only; center the training
  kernel and transform each query using those training statistics. Center the
  first ten targets and restore their mean to retain an unpenalized intercept.
- Clip predictions to [0,100], enforce a non-decreasing curve, and set the last
  support to 100, matching linear ridge. Use the existing 24-soil grouped
  evaluator and the same 21 paired-camera diagnostics.
- Compare against linear ridge and the submitted multi-crop blend. Save separate
  predictions, per-soil changes, a candidate submission, and input fingerprints.
  No crop changes, feature additions, or model blends in this comparison.
- Submit once only if both local EMD and camera disagreement improve over the
  submitted blend (40.11781628387224 and 31.52594710152303), with different test
  predictions. Otherwise record the result without a new Kaggle submission.
- Review, verify, merge into `main`, push, and remove the completed branch using
  the personal Git identity.

The fixed settings follow the documented defaults for
[kernel ridge regularization](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
and [RBF gamma](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html).
[Kernel centering](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KernelCenterer.html)
provides the centered feature-space representation. The implementation uses
NumPy; no new package or external training data is needed. Kernel and linear
ridge penalties act on different representations, so their alpha values do not
imply matched effective regularization.

Result: local EMD 52.5486 and paired-camera disagreement 25.4766. The accuracy
criterion failed, so retain the submitted blend and make no new submission.
See [kernel-ridge results](kernel-ridge-results.md). Next: confirm pretrained
weight/external-data rules, then declare one frozen image-encoder comparison.

## Frozen ResNet-18 — fixed comparison complete

Declared on 2026-09-10 before extracting features or evaluating this model:

- Use torchvision `ResNet18_Weights.IMAGENET1K_V1`, remove the classifier, and
  keep all encoder parameters frozen in evaluation mode. Extract 512 pooled
  features on CPU under inference mode; no fine-tuning or batch-normalization
  updates, including during test-image extraction.
- Reuse the existing calibrated 100 mm center crop resized to 256 × 256 with
  Lanczos. Normalize RGB with the weight recipe's ImageNet mean/std. Preserve
  the full physical crop instead of applying the recipe's additional 224-pixel
  center crop; this is a deliberate preprocessing difference from that recipe.
- Average features equally across each soil's photos and use the existing
  linear ridge predictor with fixed alpha 10. Fit feature scaling and ridge
  only on the other 23 soils in every outer fold. No PCA, feature selection,
  hyperparameter search, crop search, or blending in this comparison.
- Report all 24 held-out soils, the 21 paired-camera diagnostic soils, per-soil
  changes, and a valid ten-row candidate. Record input and weight hashes plus
  package versions; preserve prior artifacts. Keep vision packages optional.
- Submit once if both local EMD and camera disagreement improve on the submitted
  blend (40.11781628387224 and 31.52594710152303) and test predictions differ.
  Otherwise record the result without submitting. Review and test before
  merging to `main`, pushing, and deleting the completed branch.

The [official competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules),
retrieved through Kaggle's authenticated competition-page API on 2026-09-10,
allow external models unless the host specifically prohibits them (§2.6.b),
with public-access conditions for external data (§2.6.a). The freely available
[torchvision ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
fit that allowance. The full read-only receipt stays in ignored artifacts at
`artifacts/reports/pretrained_rules_official_2026-09-10.json`.

Result: local EMD 49.6530 and paired-camera disagreement 46.9890. Both submission
criteria failed, so retain the current Kaggle submission. See
[frozen-feature results](frozen-resnet-results.md). Next: declare a small ridge
regularization grid and select alpha only inside each outer training fold,
keeping the cached encoder features and physical crops fixed.

## Nested ridge regularization — complete

Declared on 2026-09-10 before computing this comparison:

- Reuse the cached 512 frozen ResNet-18 features and equal photo averaging.
  Compare only ridge penalties **10, 100, and 1000**. Keep crops, encoder,
  scaling, intercept, and cumulative-curve repair unchanged.
- Hold out each of the 24 physical soils. On its other 23 soils, choose alpha
  by mean leave-one-soil-out EMD, fitting scaling and ridge afresh on the 22
  inner training soils. Break exact score ties in favor of the larger alpha.
  Refit on all 23 outer training soils and use that same alpha for both the
  held-out soil's pooled prediction and its separate camera predictions.
- Report the outer held-out EMD and paired-camera disagreement as validation
  diagnostics. Record every inner alpha score and chosen alpha. For the final
  test candidate, select alpha by leave-one-soil-out EMD on all 24 training
  soils and refit on all 24. Its tuning score is a selection score, not another
  validation estimate. Test features and labels never choose alpha.
- Use the mathematically equivalent dual ridge solve when there are more
  features than training soils, with a numerical equivalence test. This keeps
  the nested calculation small without changing the model or adding packages.
- Save separate artifacts and input hashes; preserve previous files. Submit
  once only if outer EMD and camera disagreement both improve on the submitted
  blend (40.11781628387224 and 31.52594710152303), and test predictions differ.
  Repeated model comparisons remain exploratory even with nested tuning.
- Review and test, integrate into `main`, push with the personal Git identity,
  and delete the completed branch.

Result: outer EMD 49.9199 and paired-camera disagreement 21.1910. Camera
sensitivity improved, but accuracy failed the submission criterion. Keep the
current submission. See [nested-ridge results](nested-ridge-results.md) for
selection details and checks. Next: one fixed 50/50 blend of the submitted
multi-crop predictions and nested ridge, with aligned outer folds and camera
views, the same submission criteria, and no weight search.

## Fixed model blend and nested neighbor count — complete

Declared on 2026-09-10 before computing either comparison:

1. Average the submitted grayscale multi-crop curve and nested ResNet-ridge
   curve with fixed 50/50 weights. Reuse saved outer held-out predictions,
   separate camera predictions, and final test candidates. Align full soil or
   soil/camera keys, require identical coverage, and recompute EMD from the
   averaged curves. Do not search weights or refit either component.
2. On the original grayscale 50/100/150 mm feature caches, compare neighbor
   counts **1, 3, 5, and 7**. Choose one shared count for all three crops by
   mean inner leave-one-soil-out EMD of their equal prediction blend. Within
   each outer fold, select using only the other 23 soils; each inner fit uses
   22 soils and fits its own feature scaling. Exact ties prefer the larger
   count. Refit on the 23 outer training soils and use that same chosen count
   for the held-out soil's pooled and separate camera predictions. Keep equal
   photo averaging, uniform neighbor weights, crop sizes, and features fixed.
3. For the neighbor-count test candidate, select on all 24 labeled soils using
   inner leave-one-soil-out scoring, then refit on all 24. Record that score as
   a selection score; use the outer held-out score for validation. Record every
   candidate count's score and the chosen count in each fold. Test data never
   selects the count. Verify that fixed count 3 reproduces the existing blend.

Both comparisons write separate artifacts, per-soil changes, and input hashes,
preserving earlier outputs. Compare against the original submitted crop blend
(40.11781628387224 EMD; 31.52594710152303 camera disagreement). A candidate is
eligible for one Kaggle submission only if it improves both measures and changes
test predictions. If an earlier candidate is promoted during this batch, a
later submission must also improve both measures over that new incumbent.
Do not select settings using the public score. These remain exploratory model
comparisons; no soils are excluded because of their errors.

Commit and push tested milestones with the personal Git identity, then merge
completed work into `main` and remove the development branch. Spatial coverage
and new feature/model comparisons remain separate later experiments.

Results: the fixed blend scores 43.7447 EMD and 20.6841 camera disagreement;
accuracy fails the promotion criterion. Nested neighbor selection chooses
three in all 24 outer folds and for the final candidate, reproducing 40.1178
EMD and 31.5259 camera disagreement with unchanged test predictions. Neither
candidate is submitted. See [the comparison report](blend-and-neighbors-results.md).
Next: declare fixed spatial crop locations and aggregation, keeping the
grayscale features, crop sizes, three neighbors, and whole-soil validation.

## First experiment batch: fixed before seeing results

| Experiment | Image features | Crop | Predictor |
| --- | --- | --- | --- |
| Reference | Existing 13 color/texture features | 100 mm | 3 nearest samples |
| Grayscale | Gray 10th/50th/90th percentiles + 4 texture features | 100 mm | 3 nearest samples |
| Normalized grayscale | Per-image 5th–95th percentile contrast stretch, then grayscale features | 100 mm | 3 nearest samples |
| Smaller crop | Grayscale features | 50 mm | 3 nearest samples |
| Larger crop | Grayscale features | 150 mm | 3 nearest samples |
| Ridge | Existing 13 color/texture features | 100 mm | Ridge, fixed alpha = 10 |

All crops use native-to-downloaded PPM correction and render at 256 × 256 pixels.
Average features across photos to give each soil one supervised row. Fit scaling
and regression only on the other 23 soils in each outer fold. Ridge uses an
unpenalized intercept, then clips outputs to [0,100], enforces a non-decreasing
curve, and fixes the final value to 100.

For the camera check, query each camera's photos of the held-out soil separately;
all photos of that soil remain excluded from training. Report the disagreement
between its camera predictions alongside their errors. This is a sensitivity
diagnostic, not an estimate of performance on the unseen iPhone cameras.

These six comparisons are exploratory. Do not call the best score an unbiased
estimate of a tuned model. No setting is selected or tuned using outer-fold
labels. If we tune parameters later, use an inner validation loop within each
outer training fold. Keep the current baseline and submission files unchanged.

## Competition work alongside modeling

- Checkpoint the current code and keep an experiment table with settings,
  per-sample errors, and camera sensitivity.
- Preserve valid grain curves and physical-sample separation in every model.
- Investigate the largest errors and near-duplicate scenes without dropping
  difficult samples merely because they hurt the score.
- Make a deliberate first Kaggle submission when ready; record the public score
  as an external check, and avoid tuning repeatedly against it.
- Confirm competition deadlines and pretrained/external-data rules before those
  affect an experiment. None of the first-batch experiments needs external data.

## First-batch implementation and acceptance

- [x] Commit the verified existing baseline and this roadmap (`a4da8c7`).
- [x] Add focused tests for ridge scaling, curve validity, and held-out isolation;
  implement a small NumPy ridge predictor without adding dependencies.
- [x] Add grayscale/contrast/crop controls with defaults that reproduce the old
  features, plus regression tests for their actual image behavior.
- [x] Add one fixed experiment command that writes scores, held-out predictions,
  camera diagnostics, and separately named candidate submissions under artifacts.
- [x] Run the full batch, validate outputs, and verify the original baseline files
  and 45.24084370102147 score remain unchanged.
- [x] Review the implementation, record results in `docs/first-experiments.md`,
  and recommend the next step based on both accuracy and camera sensitivity.
