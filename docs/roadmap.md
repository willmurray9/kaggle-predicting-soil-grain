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
3. **Kernel ridge — later, if justified.** Test a small nonlinear model if the
   first comparisons suggest the relationship needs more flexibility.
4. **Frozen pretrained image features + ridge — later.** Confirm the competition's
   pretrained-weight rules, then try one visual encoder with sample-level feature
   aggregation. Keep the encoder frozen initially.
5. **Light fine-tuning — optional.** Adapt a small part of the encoder only if
   frozen features show stable value. Training a large network from scratch is
   not a priority with 24 labeled samples.
6. **Blend complementary models — optional.** Consider a simple blend only when
   held-out predictions show that the models make different errors.

Results and the next decision are recorded in [first-experiments.md](first-experiments.md).

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
