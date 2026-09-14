# Nested kernel ridge on RGB + texture — 14 September 2026

The six-setting nested search improves the fixed kernel model from **52.95787
to 48.58500 local EMD**, but remains worse than the same-input linear ridge
incumbent at **43.40571**. Camera disagreement also worsens to **31.60399** versus
29.11306. The nested candidate fails all four submission criteria; **no upload
is made**. RGB + texture remains best publicly at **61.11357**, with one of
today's five slots used.

The [declaration](roadmap.md#nested-kernel-ridge-on-rgb--texture--declared-2026-09-14)
was committed as `1b93cda` before predictions or scores. Producing code is
`fa597fdf8013f0cad756016feb42be69af3d72d3`. The fixed kernel is diagnostic only;
the nested procedure is the sole upload candidate. No grid expansion or blend
was added after the result.

## What was held fixed

Use the incumbent's 17 float64 features: the original 13 RGB/texture descriptors
plus four contrast-normalized physical texture measurements. Align both photo
caches by split, soil, camera, and path; average photos equally within each soil.
The images, 100 mm crop, calibration, targets, and curve repair stay unchanged.
The alpha-10 linear ridge reference reconstructs its saved OOF, camera, and test
predictions within **1.14e-13** before any new kernel evaluation.

Reuse the existing NumPy RBF kernel-ridge solver, with train-only feature
standardization, centered kernels, and an unpenalized intercept. Compare alpha
**0.1, 1, 10** with gamma **0.1/17, 1/17**. Lower gamma gives broader similarity
on standardized features. These parameter meanings follow the official
[kernel-ridge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
and [RBF](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html)
definitions; the implementation adds no dependency.

Each outer fold holds out all photos of one soil. Select one shared parameter
pair by inner whole-soil LOO EMD on the other 23 soils, refitting scaling and
kernel centering on each set of 22 inner training soils. Exact ties prefer larger
alpha, then smaller gamma. Refit the pair on the 23 outer training soils for
both pooled and separate-camera predictions. Select again using all 24 training
soils for the final test fit. Test features never select parameters.

## Local results and decision

| Model on the same 17 features | Outer EMD, 24 soils | Camera disagreement, 21 pairs |
| --- | ---: | ---: |
| Linear ridge incumbent, alpha 10 | **43.40571** | 29.11306 |
| Fixed kernel, alpha 1 / gamma 1/17 | 52.95787 | **24.30804** |
| Nested kernel, six settings | 48.58500 | 31.60399 |

Tuning improves 16 soils and worsens eight relative to the fixed kernel. Against
linear ridge, however, it improves only 11 and worsens 13. Its mean error rises
by **5.17930 EMD (11.93%)**.

| Submission criterion versus incumbent | Nested result | Decision |
| --- | ---: | --- |
| Mean gain ≥ 1 EMD | −5.17930 | Fail |
| At least 12/24 soils improve | 11 | Fail |
| Camera disagreement ≤ 29.11306 | 31.60399 | Fail |
| Positive mean gain without largest beneficiary | −5.79056 | Fail |

H374 worsens by **55.33402 EMD**, F827 by **53.35165**, and H666 by **22.53349**.
The largest improvement is H372, at 8.87989 EMD. Even omitting the single largest
regression, H374, from a sensitivity calculation leaves a 2.99866 EMD average
loss. No soil is removed from fitting or primary evaluation.

## What the failure tells us

The errors run in both directions. For F827 at 0.063 mm, the true cumulative
percentage is 89.56; linear ridge predicts 51.79 and nested kernel only 30.36.
For H374 at 2 mm, truth is 29.42; linear ridge predicts 31.94 and kernel 58.70.
The more flexible model misses both a fine and a coarser distribution.

A simple claim that these soils are far from every training example is not
supported. In their selected outer-fold RBF representations, maximum training
similarity is about 0.93 for F827 and 0.63 for H374, on a scale with identical
feature vectors scoring 1. These values use each fold's training-only scaler and
selected gamma. F827's nearest feature neighbor is H030. Similarity in these
compact descriptors does not guarantee similarity in grain curves.

Limited target information in the descriptors, the small number of labeled
soils, and the model's inductive assumptions are plausible explanations, not
established causes. This comparison rejects this bounded kernel procedure for
submission; it does not establish that every nonlinear model will fail.

## Parameter selection

Every outer fold chooses alpha **0.1**. Gamma is **0.1/17 for 23 folds**, and
**1/17 only for F827**. Final selection chooses **alpha 0.1 / gamma 0.1/17**.
The table below contains full-training selection scores, **not outer validation
estimates**:

| Alpha | Gamma scale (gamma × 17) | Selection EMD |
| ---: | ---: | ---: |
| 0.1 | 0.1 | **46.65760** |
| 0.1 | 1 | 49.84587 |
| 1 | 0.1 | 55.97665 |
| 1 | 1 | 52.95787 |
| 10 | 0.1 | 81.12518 |
| 10 | 1 | 73.56608 |

Selection reaches the low-alpha/broad-kernel edge of the declared grid. That is
a recorded limitation, not permission to extend this round's search after seeing
the scores.

## Reproduction and verification

Run `make texture-kernel` after the existing first-batch and physical-texture
experiments. It writes the combined photo cache, fixed/nested OOF and camera
curves, all inner/final choices, per-soil comparisons, and the submission decision
to `artifacts/experiments/nested_texture_kernel/`. It never uploads to Kaggle.

- All **169 tests pass**, with both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34895879888) passing. Coverage includes nondefault gamma against an independent
  intercept-system solution, unchanged defaults, inner/outer isolation, camera
  reuse, cache alignment, corrupted-input rejection, and test-data independence.
- A controlled test confirms that a perfect fixed diagnostic cannot authorize a
  failing nested candidate. Independent review found no actionable bugs; the
  output audit reproduces metrics, parameter selections, and the no-upload decision.
- All **122 previous artifacts remain byte-identical**. The manifest fingerprints
  ten inputs, the combined feature cache, and the valid ten-row candidate. The
  original texture manifest's 172 source hashes also match.

Unsubmitted candidate: `artifacts/submissions/rgb_texture_nested_kernel.csv`.
SHA-256: `4c24b6a3787bcbb1aa9d4057626515a23b2dec2e1b5fd83a0bb61d148c8f1f32`.

Next: a bounded linear support-vector regression comparison on these same inputs.
Its [epsilon-insensitive loss](https://scikit-learn.org/stable/modules/svm.html#svr)
grows linearly outside a tolerance, making it closer to the competition's
absolute-error metric than ridge's squared loss. It is not identical to EMD and
may still lose; declare its settings and use the same grouped validation before
judging it. Keep the current kernel grid and candidate unchanged.
