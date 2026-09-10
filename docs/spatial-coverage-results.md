# Fixed spatial coverage — 2026-09-10

Five positions per photo produce **42.53062 local EMD**, versus **40.11782**
for the submitted center-only crop blend. Paired-camera disagreement improves
from **31.52595 to 25.20834**. Broader coverage helped camera agreement but did
not improve local accuracy. Lower is better for both measures.
The planned Kaggle submission completed at **82.91954 public EMD**, worse than
the original blend's **76.75797**. It does not replace RGB ridge, our current
best public submission at **61.87967**.

The [declaration](roadmap.md#spatial-coverage-and-informative-submissions--declared-2026-09-10)
was committed as `5962063` before scoring this experiment or making today's
informative submissions. This candidate is the third planned upload, regardless
of its local ranking; [public results and provenance](submissions.md) are recorded
separately. No positions, weights, or neighbor counts were tuned.

## What changed

At each of the existing 50, 100, and 150 mm crop sizes, extract the center plus
four positions at quarter/three-quarter fractions of the available horizontal
and vertical crop-origin travel. All patches stay within the oriented image.
The original center crop remains exactly unchanged. The largest crop leaves only
3.26 mm of half-margin on the narrowest Motorola views; fixed 25 mm shifts would
not fit. Consequently the physical displacement varies across fields of view.

Each patch retains the same seven grayscale statistics and 256-pixel rendering.
Average features equally across five patches per photo, then equally across
photos per soil. Each crop model still averages three nearest training curves;
the final curve averages the three crop models equally. The feature dimension
remains seven, and the number of independent training soils remains 24.

All photos and patches of the held-out soil stay out of training. Feature scaling
is fitted on the other 23 soils. Separate held-out camera predictions use those
same training soils. The experiment processes 162 photos and 2,430 patches.

## Results

| Crop/model | Center-only local EMD | Five-position local EMD | Center-only camera disagreement | Five-position camera disagreement |
| --- | ---: | ---: | ---: | ---: |
| Gray 50 mm | 39.91976 | 49.35941 | 43.48283 | 27.65320 |
| Gray 100 mm | 42.98204 | 40.67935 | 33.07691 | 31.19642 |
| Gray 150 mm | 41.59825 | 41.70384 | 32.66882 | 30.74565 |
| Equal curve blend | **40.11782** | **42.53062** | **31.52595** | **25.20834** |

The blended local error rises 6.0%; camera disagreement falls 20.0% across the
same 21 camera pairs. Nine soils improve, fourteen worsen, and one is unchanged
within 1e-9. All ten test predictions change relative to the original submission.

| Soil | Center-only EMD | Spatial EMD | Improvement (positive is better) |
| --- | ---: | ---: | ---: |
| H374 | 50.7530 | 31.0730 | +19.6800 |
| G190 | 38.9997 | 24.4428 | +14.5568 |
| H666 | 68.7636 | 59.2579 | +9.5057 |
| H126 | 35.0151 | 58.5736 | −23.5585 |
| H037 | 44.9617 | 65.7584 | −20.7967 |
| H038 | 104.8370 | 122.6234 | −17.7864 |

The 50 mm component loses the most accuracy despite substantially better camera
agreement. Averaging spatial features may reduce variation between views while
also removing useful local distinctions. That is a hypothesis, not a demonstrated
cause. The opposite direction at 100 mm shows that wider coverage does not have
one uniform effect. We retain the fixed blend for the declared submission rather
than selecting a new component or weight from these results.

## Reproduction and checks

Run `make spatial-coverage` after `make data` and `make multicrop`. It writes three
photo-feature caches, component and blended held-out/camera predictions,
per-soil comparisons, and a manifest under `artifacts/experiments/spatial_coverage/`.
The manifest fingerprints the configuration, calibration, labels, template,
photo index, reference predictions, every input photo, and the candidate CSV.

- Candidate: `artifacts/submissions/spatial_multicrop.csv`.
- SHA-256: `9f9b9f20135404110d2fcb6910b3580977dd6549ad80269947ee50d5c36c1e41`.
- All 98 tests and both remote CI jobs pass, including calibrated shifted crops, EXIF orientation,
  unchanged default pixels, equal patch/photo weighting, held-out isolation,
  and output alignment. Independent review found no actionable issues.
- All 82 pre-existing artifact files were checked and remain byte-identical.
- Candidate passes the submission validator and changes all ten test curves
  relative to each previously submitted model.
- Fifteen original center-crop feature checks spanning all five cameras and
  three crop sizes reproduce their saved features within 9.72e-17.

## Next comparison

The next roadmap step is a small set of texture measurements at fixed physical
scales, with compact inputs and grouped validation. Keep the original crop blend
as the historical local reference and RGB ridge as a useful regression reference.
First declare a bounded texture feature set and one fixed predictor; retain the
same inputs and folds so the effect is attributable to the feature change.
Today's three public submissions are a completed comparison batch, not a reason
to start searching leaderboard-driven blend weights.
