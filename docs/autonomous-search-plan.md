# Autonomous model search — September 16, 2026

The user authorized autonomous experiments and submissions until a public score
beats **55.78511**, or today's submission allowance is exhausted. The fresh API
check at 18:13 UTC shows **0 of 5** submissions used today. This replaces the
earlier requirement to improve both local EMD and camera consistency before
uploading. Those measurements remain useful diagnostics, not upload vetoes.

## Fixed batch (declared before computing new scores)

1. Average the incumbent spectral ridge and saved DINO PCA/ridge CDFs 50/50.
2. Average the incumbent and saved MobileNet PCA/ridge CDFs 50/50.
3. Fit supervised PLS to the cached 384 DINO features. Preserve float32 photo
   averaging, then fit in float64. Standardize features within each training
   fold (population standard deviation; constant columns use 1). Jointly fit
   the first ten raw percentage targets with `PLSRegression(scale=False,
   max_iter=500, tol=1e-6)`. Select 1, 2 or 4 components by inner whole-soil LOO
   EMD; exact ties choose fewer components. No preliminary PCA or target scaling.
4. Fit ten independent gradient-boosted regressors to the 23 spectral features,
   averaged equally over each soil's photos in float64. Fixed settings: 128
   stages, learning rate 0.03, depth 2, minimum 4 soils per leaf, squared-error
   loss, all features and rows, random seed 42, no early stopping or search.
5. Reuse the nested ridge evaluator on the same 23 features, selecting alpha
   from 10, 100 and 1000 within every training fold; exact ties favor larger alpha.
6. Include the existing unsubmitted DINO PCA/ridge and MobileNet PCA/ridge
   standalones as two further candidates, without changing their predictions.

All model fits exclude the entire held-out soil. Nested parameters and learned
transforms are fitted inside every relevant fold. Final models use all 24
training soils and no test labels. Predictions use the existing clipping,
cumulative-maximum and final-100 repair. Blend predictions align by soil and,
for camera diagnostics, by soil/camera; weights are never searched.

## Submission sequence and stopping

Complete this batch before uploading. Order the seven candidates by local
whole-soil EMD, then camera disagreement, then experiment name. Skip any CSV
whose aligned predictions duplicate a prior upload or an earlier queued
candidate (absolute tolerance 1e-9). Submit sequentially, at most five today,
checking fresh capacity and the exact returned reference before continuing.
Stop immediately if a completed score is below 55.78511. Do not use intervening
public results to alter models, weights or the queue. If a candidate fails a
numerical/convergence check, record its exclusion rather than silently changing
its recipe. The batch contains enough distinct model families for five attempts;
if fewer remain valid, declare any additional experiment before evaluating it.

Local ranking is exploratory after many experiments on only 24 soils. Nested
validation protects the declared tuning step, but does not erase selection bias
across this entire competition project. Public scores are separate observations.

## Reproducibility and Git

Verify saved source/cache hashes and reconstruct the spectral reference before
new scoring. Preserve all 210 existing artifacts byte-for-byte. Write this batch
to a new directory, recording input/output hashes, versions, selection tables,
per-soil errors and the producing commit. Test the model math and fold boundaries;
run the complete suite, review and push the producing code before execution.
Require successful code CI and validated template-aligned CDFs before upload.
Persist a receipt before each request; resolve uncertain responses through
history rather than blindly retrying. Commit results with Will Murray's personal
email, merge completed work into main, push, and delete the merged branch.

The PLS and boosting implementations follow the official scikit-learn
[cross-decomposition guide](https://scikit-learn.org/stable/modules/cross_decomposition.html)
and [gradient boosting API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
