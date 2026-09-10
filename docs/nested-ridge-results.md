# Nested ridge regularization on frozen features

Choosing regularization inside each training fold did not improve the submitted
model's accuracy. It reduced sensitivity to camera views, but the candidate
fails the predeclared accuracy criterion and was **not submitted**.

| Model | Local EMD | Paired-camera disagreement EMD |
| --- | ---: | ---: |
| Submitted grayscale multi-crop blend | 40.1178 | 31.5259 |
| Frozen ResNet-18, fixed alpha 10 | 49.6530 | 46.9890 |
| **Frozen ResNet-18, nested alpha selection** | **49.9199** | **21.1910** |

Lower is better. All 24 physical soils and the same 21 paired-camera soils are
included. The nested model improves six soils and worsens 18 relative to the
submitted blend. F827 improves by 36.25 EMD, H374 by 30.33, and H493 by 27.75;
G190 worsens by 55.95 and H405 by 49.44. Relative to fixed-alpha frozen ridge,
13 soils improve and 11 worsen, with slightly worse mean error overall.

## Selection and validation

The grid **10, 100, 1000** and submission rule were committed in `9c5a549` before
running the comparison. For every outer held-out soil, leave-one-soil-out
validation on the remaining 23 chooses alpha. Every inner fit standardizes
using its 22 training soils only. Exact score ties prefer the larger alpha.
The chosen model is refitted on all 23 outer training soils and predicts both
the pooled held-out soil and its individual camera views. Outer folds selected
alpha 1000 for 16 soils and alpha 100 for eight.

The final test model selects alpha using all 24 training soils, then refits on
all 24. Its selection scores are:

| Alpha | Full-training LOO selection EMD |
| ---: | ---: |
| 10 | 49.6530 |
| 100 | 46.1003 |
| 1000 | 45.4591 |

Alpha 1000 is selected for the final candidate. **45.4591 is a tuning score,
not the nested validation result**; the outer estimate is 49.9199. Neither
test features nor public leaderboard results choose alpha. Nested tuning
separates parameter selection from outer evaluation, but our repeated model
comparisons still make the overall search exploratory. Camera disagreement
does not measure performance on the unseen iPhone models.

## Reproduction and checks

After `make frozen-model` has written the image-feature cache:

```bash
make nested-ridge
make validate SUBMISSION=artifacts/submissions/frozen_resnet18_nested_ridge.csv
make test
```

This command needs no vision packages, checkpoint download, or new feature
extraction. It restores cached feature columns as float32, preserving the
original encoder's photo averaging. The ridge predictor uses the equivalent
dual solve when features outnumber training soils. The reconstructed fixed-alpha
pooled and camera predictions agree with the previous outputs within 5e-13.

`artifacts/experiments/nested_ridge/` contains outer predictions, camera
predictions, all 72 inner alpha scores and selections, the three final selection
scores, per-soil comparisons, and a summary. The summary records the encoder
metadata and hashes of the cache, its source manifest, labels, template, config,
and reference predictions. All 62 previous artifacts remain byte-for-byte
unchanged. Generated outputs remain outside Git.

The ten-row candidate passes submission validation and differs from the
submitted blend on every test soil. Its SHA-256 is
`cb62d4b0c8e03a8e306e697900cedcd00379ac6751d77d2d0a549e83c1a85838`.
All 63 tests pass, including held-out label isolation, inner-fold refitting,
camera consistency, test-data independence, ID alignment, provenance, and
primal/dual numerical equivalence. Independent code review found no issues.

## Next comparison

The frozen model still helps some difficult soils and its regularized version
has lower camera disagreement. Test one fixed 50/50 prediction blend of the
submitted multi-crop model and nested ridge, aligning saved outer-fold soil and
camera IDs. Use the same submission criteria and no weight search. This tests
whether their errors offset each other before investing in encoder fine-tuning.
