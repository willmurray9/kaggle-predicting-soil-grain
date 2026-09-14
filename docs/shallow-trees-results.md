# Fixed shallow ExtraTrees — 14 September 2026

The fixed tree ensemble scores **47.52880 local EMD**, worse than the same-input
ridge incumbent's **43.40571**. Camera disagreement improves to **25.45697** from
**29.11306**, but only **11/24 soils** improve. Three of the four submission
criteria fail, so **no candidate is uploaded**. RGB + texture retains the best
public score of **61.11357**; today's submission count remains one of five.

The [declaration](roadmap.md#fixed-shallow-extratrees--declared-2026-09-14) was
committed as `f733c39` before scoring. Producing code is
`491d8ceb91a9ab7694d94f2ef4c1ddd604e024bb`. No settings or submission criteria
changed after the result.

## Controlled comparison

Reuse the incumbent's exact 17 RGB + physical-texture features, calibrated
100 mm center crops, and equal float64 photo means per soil. Verify the recorded
source hashes and photo keys, then reconstruct the saved ridge OOF, camera, and
test curves. The largest reconstruction difference is **1.14e-13**.

Fit one native ten-output `ExtraTreesRegressor`: **256 trees**, **maximum depth
3**, **minimum 3 soils per leaf**, all 17 features available at each split,
squared-error criterion, no bootstrap, seed 42, and one worker. Each tree shares
its splits across the ten targets; terminal leaves average training curves and
the forest averages the trees. Keep the existing clipping, monotone repair, and
final 100% endpoint. See the [official estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

There is no scaling, PCA, tuning grid, seed search, or blend. Scikit-learn
internally converts feature matrices to float32; photo aggregation remains
float64. Every outer fold excludes all photos of one soil and trains on the
other 23 soil means. The existing evaluator refits for pooled and camera queries;
identical training rows and the fixed seed yield identical forests. A final fit
on all 24 soils produces the ten-row test candidate. No inner selection is
needed because the configuration was fixed beforehand.

## Results and submission screen

| Model on the same 17 inputs | Outer EMD, 24 soils | Camera disagreement, 21 pairs |
| --- | ---: | ---: |
| Ridge incumbent, alpha 10 | **43.40571** | 29.11306 |
| Fixed shallow ExtraTrees | 47.52880 | **25.45697** |

| Criterion versus ridge incumbent | Tree result | Decision |
| --- | ---: | --- |
| Mean gain ≥ 1 EMD | −4.12310 | Fail |
| At least 12/24 soils improve | 11 improve, 13 worsen | Fail |
| Camera disagreement ≤ 29.11306 | 25.45697 | Pass |
| Positive gain without largest beneficiary | −5.66402 | Fail |

The largest losses are **H374 (+76.78762 error)**, G190 (+30.50705), F827
(+24.01405), and H666 (+20.59189). The strongest gains are H366 (−31.31805
error), H549 (−17.95580), and H126 (−14.15722). H374 accounts for about 77% of
the net error increase summed across all soils. All soils remain in fitting
and the primary score; omitting the largest beneficiary is only the declared
sensitivity calculation.

The predictions show substantial averaging toward less extreme curves. At
0.63 mm, H374's measured fraction is **18.25%**, ridge predicts **16.67%**, and
the trees predict **59.95%**. For the very fine F827 soil, **89.56%** passes
0.063 mm, versus **51.79%** from ridge and **41.52%** from trees. This is
consistent with the shallow, minimum-leaf averaging being too restrictive for
some soils; it does not isolate depth, features, or target coverage as the cause.
Trees also cannot extrapolate beyond the training target ranges.

Lower camera disagreement alone does not establish a better predictor: a more
averaged prediction can be stable while less accurate. This fixed tree model
does not supply the broad accuracy gain required for a selective submission.
The result does not rule out other tree configurations, but no follow-up depth,
leaf-size, feature-subset, or blending search is part of this batch.

## Reproduction and verification

Run `make shallow-trees` after the combined cache from `make texture-kernel`
exists. The command writes separate OOF/camera curves, per-soil comparisons,
settings, source hashes, runtime versions, and the decision under
`artifacts/experiments/shallow_trees/`. It validates a separate candidate CSV
and does not upload to Kaggle. There are no new dependencies or tuning files.

- All **218 tests pass** locally, and both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34898363634)
  pass. Coverage includes analytical leaf means,
  constant features, deterministic/query-independent predictions, valid curves,
  actual depth/leaf limits, whole-soil exclusion, shuffled key alignment,
  reference reconstruction, source corruption, and input preservation.
- Independent implementation review found no actionable issues. A separate
  saved-output audit verifies metrics, coverage, provenance, and the decision.
- All **137 prior artifacts remain byte-identical**. The manifest fingerprints
  11 inputs and the valid candidate. This run used Python 3.11.13, NumPy 2.4.6,
  pandas 3.0.3, SciPy 1.17.1, and scikit-learn 1.9.1.

Unsubmitted file: `artifacts/submissions/rgb_texture_shallow_trees.csv`.
SHA-256: `dacf8fd092c27430147da6a792df198bf759440cb1ea9182441a2264a9e52ba9`.

Next: return to the planned representation comparison with a second frozen
image encoder, declaring its preprocessing and compact regression head before
scoring. Kernel ridge, linear SVR, and this fixed tree model have not supplied
a qualifying improvement on the same 17 inputs. A second encoder tests whether
different image information helps; it is not evidence that fine-tuning is needed.
