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
