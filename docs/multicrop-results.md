# Equal-weight multi-crop result

The fixed blend averages the existing grayscale 50, 100, and 150 mm 3-neighbor
predictions. Each crop has one-third weight. No parameters were tuned. The
[submission rule](roadmap.md#current-follow-up-error-audit-and-equal-weight-multi-crop-blend)
was recorded in commit `2f512cd` before computing this result.

| Model | Leave-one-soil-out EMD | Paired-camera disagreement EMD |
| --- | ---: | ---: |
| Original RGB 100 mm reference | 45.2408 | 33.6083 |
| Ridge RGB 100 mm | 41.2059 | 28.8910 |
| Gray 50 mm | 39.9198 | 43.4828 |
| Gray 100 mm | 42.9820 | 33.0769 |
| Gray 150 mm | 41.5983 | 32.6688 |
| **Equal-weight multi-crop** | **40.1178** | **31.5259** |

Lower is better. Validation covers all 24 physical soils; camera disagreement
covers the 21 soils with paired cameras. Blending aligns held-out predictions by
soil ID and camera predictions by soil ID plus camera, then recomputes EMD from
the averaged curves. It never averages scores. All photos of the held-out soil
remain excluded from training in each component.

The blend meets both predeclared criteria against the original reference and is
the candidate for the first Kaggle submission. Gray 50 mm retains the lowest
local error, while ridge retains lower camera disagreement than the blend.
Neither property alone was our submission rule.

Against the original reference, the blend improves 13 soils, ties one, and
worsens ten. The largest improvements are H374 (75.78 EMD) and H405 (48.73);
the largest regression is G190 (24.37). This is a small exploratory comparison,
not evidence of uniform improvement or an unbiased estimate of performance on
the unseen test-set iPhones.

The [visual error audit](error-audit.md) found limitations in the summary
features and visible differences between scenes, without evidence to change
labels or exclude samples. All five original baseline artifacts remain
byte-for-byte unchanged.

## Reproduce

After preparing data and running the first experiment batch:

```bash
make multicrop
make audit
make validate SUBMISSION=artifacts/submissions/multicrop_baseline.csv
make test
```

The blend writes aligned OOF and camera predictions plus `summary.json` under
`artifacts/experiments/multicrop/`. The summary fingerprints the five component
prediction files with SHA-256. The submission is
`artifacts/submissions/multicrop_baseline.csv`, with ten rows in the official
template's order and valid cumulative curves. Generated outputs remain ignored
by Git.

Validation: all 37 tests pass, including shuffled-ID alignment, duplicate and
missing component rejection, curve constraints, and scoring after averaging.
The generated ten-row candidate passes the submission validator.
