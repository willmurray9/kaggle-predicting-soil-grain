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

## Spatial coverage and informative submissions — declared 2026-09-10

Before computing spatial scores or viewing new public results:

- Extract five patches per photo at each existing 50/100/150 mm crop size:
  center plus four positions at (0.25, 0.25), (0.25, 0.75), (0.75, 0.25),
  (0.75, 0.75). Coordinates are fractions of the available crop-origin travel
  after EXIF orientation: left = floor(x × (width − side)), likewise top.
  Thus all crops fit without padding; the default center remains unchanged.
  A geometry check found only 3.26 mm of half-margin for the largest crop on
  some Motorola photos, making a common 25 mm shift infeasible. A montage of
  one image per camera was inspected without scoring positions. Physical
  displacement varies with field of view; this is a limitation of this test.
- Average the same seven grayscale features across the five patches within
  each photo, then equally across photos within a soil. Keep 256-pixel rendering,
  three uniform neighbors, training-fold feature scaling, and the equal blend
  across the three crop sizes. Do not tune positions or counts in this batch.
- Hold out every photo and patch of one physical soil together. Save component
  and blended OOF/camera predictions, per-soil changes, photo features, source
  hashes, and a separately named spatial submission. Preserve previous files.
- Test crop location/calibration, unchanged default behavior, equal patch/photo
  weighting, whole-soil exclusion, and submission alignment; run the full suite.

The user's new instruction supersedes the earlier requirement that submissions
beat both local metrics. Use up to **three informative submissions** in this
batch, selected now, irrespective of their relative local scores:

1. Existing RGB ridge (fixed alpha 10): does learning a linear mapping from the
   original simple features transfer better than nearest-neighbor averaging?
2. Existing nested ResNet ridge: does the learned representation and stronger
   regularization transfer better despite its worse outer validation score?
3. New five-position grayscale multi-crop: does broader spatial coverage help
   on the unseen test cameras?

Submit each valid, numerically distinct candidate once after validating its
provenance and recording its producing code. Do not change this batch based on
intermediate public scores, search blend weights, or resubmit the duplicate
nested-neighbor candidate. Public results are external comparisons, not unbiased
validation or a basis for declaring a private-test winner. Record all outcomes,
including regressions. Keep the original submission intact.

At 2026-09-10 22:31 UTC, authenticated Kaggle metadata reported a daily limit of
five and submission history showed one completed upload, dated 2026-09-09.
Use the user's more conservative budget of three today. Commit and push the
declaration, then tested implementation/results; finish on clean synchronized
`main` and remove the completed branch.

Spatial result: 42.5306 local EMD and 25.2083 paired-camera disagreement; nine
soils improve, fourteen worsen, and one is unchanged. See the
[spatial report](spatial-coverage-results.md) and [submission log](submissions.md).
All three declared uploads completed: RGB ridge 61.87967 public EMD, nested
ResNet ridge 63.01764, and spatial coverage 82.91954. The original scored 76.75797;
RGB ridge is now the best public submission. No further uploads in this batch.
The next modeling step is a small, predeclared set of physical-scale texture
measurements with the same grouped evaluation.

## Physical texture with a color control — declared 2026-09-11

Before computing new validation scores or viewing any new public score, run one
small factorial comparison on the existing center 100 mm crop at 256 pixels:

| Configuration | Inputs | Predictor |
| --- | --- | --- |
| RGB reference | Existing 13 RGB/texture features | Ridge, alpha 10 |
| RGB + physical texture | Existing 13 plus four new measurements | Ridge, alpha 10 |
| Grayscale control | Existing seven grayscale features | Ridge, alpha 10 |
| Grayscale + physical texture | Existing seven plus four new measurements | Ridge, alpha 10 |

The four measurements are the mean squared grayscale difference over horizontal
and vertical offsets, divided by twice the crop's grayscale variance. Use nominal
offsets **1, 2, 4, 8 mm**, rounded to **3, 5, 10, 20 pixels** at this crop scale
(actual 1.171875, 1.953125, 3.90625, 7.8125 mm). Average the two directional means
equally; do not wrap or pad pixels. If variance is below 1e-12, return four zeros.
This descriptor measures contrast-normalized spatial variation, not particle
diameters or counts. It is invariant to unclipped affine brightness changes.
Some distances overlap existing texture lags; the new information is squared,
contrast-normalized variation, not access to a wholly new range of grain sizes.

Keep center cropping, PPM correction, EXIF orientation, equal photo averaging,
ridge regularization, and curve projection fixed. Reuse the existing RGB and
grayscale photo caches after exact split/soil/camera/path alignment and finite
feature checks. Compute new texture features from the same original photos and
calibration. Fit scaling and ridge only on the other 23 soils in each held-out
fold. Each soil's camera views and patches remain excluded together. This is a
fixed exploratory comparison, with no selection of lags or regularization.

Reconstruct the RGB reference's OOF, camera, and test predictions within 1e-10
before any upload. Preserve all older files. Save per-photo features, all four
OOF/camera predictions, paired per-soil comparisons for each feature addition
and color removal, candidate submissions, settings, and source hashes. Test
descriptor values on known patterns, constant images and brightness transforms;
test cache alignment, equal photo weighting, whole-soil exclusion, and the
reference reconstruction gate. Use the existing NumPy/Pillow stack.

Today's three submission candidates are fixed now: RGB + texture, grayscale
control, and grayscale + texture. Submit each valid, distinct candidate once
after tests, review, and code push, regardless of whether it beats local EMD or
camera disagreement. Skip numerical duplicates of existing submissions. Do not
change the batch or tune against intermediate public results. The best public
reference is RGB ridge at 61.87967; the historical crop blend remains the local
reference at 40.11782 EMD. Record all results without claiming that a public
ranking proves private-test performance.

Authenticated preflight at 2026-09-11 14:14 UTC confirmed four completed lifetime
submissions, zero today, and an API limit of five per day. Use our conservative
three-upload budget. Commit/push the declaration and tested milestones with the
personal Git identity, then merge into `main` and remove the finished branch.

Local results: RGB reference 41.20591, RGB + texture 43.40571, grayscale control
43.15793, grayscale + texture 45.34803 EMD. The RGB reference reproduces all
saved prediction sets within 5.69e-14. See the
[texture report](physical-texture-results.md). All three declared uploads completed
without changing settings: RGB + texture **61.11357** public EMD (new best),
grayscale 70.83530, grayscale + texture 71.34901. This uses three submissions
today; no further uploads in this batch. Next: controlled learned-feature
preprocessing, with dimensionality reduction fitted inside every training fold.

## Fixed PCA and selective submissions — declared 2026-09-14

The user's latest instruction supersedes automatic three-candidate submission
batches: upload only with a good reason to expect improvement over the current
public best, RGB + texture at 61.11357. Preserve slots rather than use the quota.

Before calculating new validation scores, declare two candidates:

1. **Eight-component PCA + nested ridge.** Reuse the existing 512-dimensional
   ResNet photo features and equal photo averaging. In every inner and outer
   fit, standardize from its training soils, fit SVD on those standardized rows,
   and project training/query vectors onto the first eight principal components.
   Do not whiten or restandardize the component scores. This preserves ridge's
   penalty in the retained directions; keeping all components should reproduce
   ordinary ridge. Fix eight components without searching a count.
2. **Fixed 50/50 blend of PCA ridge and RGB + texture.** Align whole-soil OOF,
   camera, and test predictions exactly before averaging the two curves. Do not
   search blend weights. This separately tests whether their errors compensate.

Keep the existing alpha grid 10/100/1000 and exact-tie preference for stronger
regularization. Each outer fold selects alpha using only the other 23 soils;
every inner PCA/scaler/regressor sees only 22 soils. Fit again on the 23 outer
training soils to predict the pooled held-out soil and its separate camera views.
Select the final alpha with all 24 training soils and refit for test predictions.
Record full-training selection scores separately from outer validation scores.
Preserve float32 photo aggregation from the frozen cache. Test labels and public
scores never fit or select the representation, alpha, or blend weight.

Reconstruct the unchanged nested 512-feature reference, including OOF, camera,
alpha selections, and final test predictions. Save new OOF/camera predictions,
per-soil comparisons, all inner/final selection scores, candidate CSVs, provenance,
and the submission decision. Preserve all previous artifacts. Tests must cover
full-rank equivalence, unchanged ridge defaults, no second component scaling,
query independence, inner-fold PCA fitting, held-out label isolation, alignment,
and the submission screen.

For this round, a candidate passes the submission screen only if, relative to
the current RGB + texture model's held-out predictions, it meets **all** of:

- Mean EMD improves by at least **1.0** (reference 43.40570546697441).
- At least **12 of 24 soils** improve by more than 1e-9.
- Paired-camera disagreement does not increase (reference 29.113056428926008).
- Mean improvement stays positive after omitting its single largest beneficiary
  from the comparison. This is a sensitivity diagnostic; no soil is excluded
  from model fitting or the primary score.

Submit **at most one** valid, distinct candidate that passes. If both pass, use
lower outer EMD, with an exact tie preferring standalone PCA. If neither passes,
submit none. These are practical evidence thresholds, not a significance test or
a guarantee of better public/private performance. Do not retry with new counts,
weights, or thresholds after seeing these results.

Authenticated preflight at 2026-09-14 19:08 UTC confirmed seven completed lifetime
uploads, zero today, and a daily API limit of five. The best submission and cached
ResNet artifacts match their recorded hashes. Commit/push tested milestones with
the personal Git identity; finish on synchronized clean `main`, deleting the
completed branch. No automatic follow-up submissions are authorized by this batch.

Local results: PCA ridge 45.02758 EMD; fixed blend 42.02843. The blend improves
14/24 soils versus RGB + texture, lowers camera disagreement to 17.90374, and
retains +0.47653 mean improvement without its largest beneficiary. Only the
blend passes the declared screen. Its single upload scored 63.33017 public EMD;
RGB + texture remains best at 61.11357. One slot was used, four preserved, and
there are no follow-up uploads in this batch. See [the PCA report](pca-ridge-results.md)
and [submission log](submissions.md). Next: a separate official encoder
preprocessing comparison, with the same grouped validation.

## Official ResNet preprocessing — declared 2026-09-14

Before extracting new features or calculating new scores, compare the official
`ResNet18_Weights.IMAGENET1K_V1.transforms()` recipe with the current encoder
preprocessing. Apply it to the same calibrated 100 mm center crop, already
rendered to 256 × 256 with Lanczos. The recipe's shorter-side resize to 256
(bilinear, antialias enabled) is therefore a no-op; its center crop retains
224 × 224 pixels, a nominal **87.5 mm** field of view. Its normalization matches
the existing ImageNet mean/std. This tests the full recipe and smaller field of
view together; it cannot separate their effects. Use the same frozen checkpoint,
CPU inference, equal float32 photo aggregation, eight PCA components, and nested
ridge grid 10/100/1000, refitting scaling/PCA/ridge in every training fold.

Evaluate exactly two candidates: official-preprocessing PCA ridge and its fixed
50/50 prediction blend with RGB + physical texture. Align whole-soil and camera
keys exactly. Compare each with its corresponding saved legacy-preprocessing
candidate (PCA 45.02758096092745 EMD / 14.669811083041479 camera disagreement;
blend 42.0284318338925 / 17.90373798556798). Keep all 24 soils and the same 21
paired-camera diagnostics. No crop, component-count, alpha-grid, or weight search.

For at most **one additional upload** in this batch, require all four incumbent
criteria above (gain >= 1 EMD over RGB + texture, at least 12 soils improve,
no worse camera disagreement, positive gain without the largest beneficiary),
plus **at least 1 EMD improvement over the corresponding legacy candidate** and
**no worse camera disagreement than that candidate**. Choose lower outer EMD
among eligible candidates; an exact tie prefers standalone PCA. Validate the
candidate and reject numerical duplicates of previous uploads. These are
conservative budget screens, not statistical significance or guarantees. If
neither qualifies, submit none; do not change the screen after seeing results.

Preserve old defaults and artifacts. Save separate features, OOF/camera curves,
per-soil comparisons, inner/final alpha scores, candidate CSVs, source/photo and
checkpoint hashes, and the decision. Test the exact official transform on a
known image, legacy behavior, inference mode, grouped evaluation, alignment, and
the strengthened screen. Review and run tests before any upload; commit/push
using the personal identity, merge to main, and remove the completed branch.

Authenticated preflight at 2026-09-14 20:31 UTC confirmed eight completed lifetime
uploads, one today, and four remaining under the API limit of five. The current
best is RGB + texture at 61.11357. All seven published competition pages remain
unchanged from the saved September 10 receipts, including external-model
permission; this comparison uses the existing verified checkpoint. See the
[official ResNet-18 recipe](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

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
