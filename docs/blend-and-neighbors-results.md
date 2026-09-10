# Fixed model blend and nested neighbor count

Both comparisons declared in `4fe86f9` are complete. The fixed 50/50 blend
worsened accuracy, and nested neighbor selection chose the existing count of
three in every outer fold. Neither warrants a new Kaggle submission.

| Model | Outer held-out EMD | Paired-camera disagreement EMD |
| --- | ---: | ---: |
| Submitted grayscale multi-crop, fixed 3 neighbors | 40.1178 | 31.5259 |
| Frozen ResNet-18 with nested ridge | 49.9199 | 21.1910 |
| Fixed 50/50 blend of those models | 43.7447 | 20.6841 |
| Multi-crop with nested neighbor selection | 40.1178 | 31.5259 |

Lower is better. Every comparison includes all 24 physical soils and the same
21 paired-camera soils. Every photo of an outer held-out soil stays outside
training and any inner parameter selection.

## Fixed 50/50 blend

Average the saved multi-crop and nested-ridge curves with equal weights, aligning
soil IDs and soil/camera pairs before averaging. No weight search, feature
extraction, or component refitting. Recompute EMD from the averaged predictions.

The blend improves six soils and worsens 18 versus the submitted model. Its
largest improvements are H374 (18.34 EMD), F827 (18.13), and H493 (13.87). Its
largest regressions are G190 (27.98), H405 (24.53), and H549 (19.75). Averaging
reduces camera disagreement, but it does not offset the accuracy regressions.
All ten test predictions change, yet the declared accuracy criterion fails.
**Do not submit this candidate.**

## Nested neighbor count

Reuse the original seven grayscale features at each of 50, 100, and 150 mm.
Keep equal photo weighting and equal crop-prediction weighting. In each outer
fold, select one shared count from **1, 3, 5, 7** for all three crops. Score the
equal crop blend in inner leave-one-soil-out validation on the other 23 soils;
each inner fit uses only its 22 training soils for scaling and neighbors.
Exact score ties prefer more neighbors. Use the chosen count for both the
outer held-out soil's pooled prediction and its camera predictions.

**All 24 outer folds select three neighbors.** Their predictions reproduce the
existing baseline within floating-point precision. This supports retaining
three within the tested grid; it does not establish that it is optimal for
other features, distance weights, or populations.

Final selection on all 24 labeled soils also chooses three:

| Shared neighbor count | Full-training LOO selection EMD |
| ---: | ---: |
| 1 | 49.5114 |
| 3 | 40.1178 |
| 5 | 43.5759 |
| 7 | 44.0148 |

These are selection scores, not four independently selected validation results.
The nested outer EMD is 40.1178. Unlike the previous ridge experiment, every
outer selection and the final selection agree with the original fixed setting.
The final candidate has no materially changed test predictions (comparison
tolerance 1e-9), so **do not submit a duplicate prediction**.

## Reproduction and verification

After `make experiments`, `make multicrop`, `make frozen-model`, and
`make nested-ridge` have produced their source artifacts:

```bash
make model-blend
make nested-neighbors
make validate SUBMISSION=artifacts/submissions/multicrop_nested_ridge_blend.csv
make validate SUBMISSION=artifacts/submissions/nested_neighbors_multicrop.csv
make test
```

The two new commands need only cached features/predictions and core dependencies;
they do not download weights or extract photographs. Separate outputs live in
`artifacts/experiments/model_blend/` and `artifacts/experiments/nested_neighbors/`.
Both contain outer predictions, camera predictions, per-soil comparisons, and
summaries with source hashes. Neighbor selection also records all 96 inner grid
scores and the four final selection scores. Both candidate files pass the
submission validator; all 69 earlier artifacts remain byte-for-byte unchanged.

Candidate SHA-256 values:

- Fixed blend: `de7cf5a71cb562d7451168d26a03e9b0fa60afc98cdd0e47c41f011d96a72f80`.
- Nested neighbors: `b6c91719556d57067ba292b4eb966cc2c9288db40fe016b7034272bcf9b6b8f2`.

The latter file can have a different byte hash from the submitted file despite
equivalent predictions, because recomputation and CSV serialization introduce
tiny numerical differences. The original submitted file is preserved exactly.

All 87 tests pass. New checks cover independent nested calculations, held-out
label/query isolation, train-only scaling, shared camera selection, full ID
alignment, averaging predictions before scoring, and source preservation.
An independent review found no actionable issues. Fixed count three reconstructs
the old pooled, camera, and test curves within 1e-12.

## Next experiment

Improve spatial coverage with a fixed set of crop locations, retaining the
grayscale features, physical crop sizes, equal averaging, and three neighbors.
This follows the audit observation that coarse particles move into and out of
center crops. Declare feasible locations and the aggregation rule before
scoring; retain whole-soil folds and the existing submission criteria. Distance
weighting, new texture measurements, and model changes remain separate tests.
