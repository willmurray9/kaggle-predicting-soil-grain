# Fixed RBF kernel ridge

The next nonlinear model uses the same 13 RGB/texture measurements, 100 mm crop,
and equal photo averaging as linear ridge. Settings were committed in `3d4f733`
before running predictions: RBF `gamma = 1/13`, `alpha = 1`, and an unpenalized
intercept. Scaling, kernel centering, and target centering use only each fold's
training soils. Output clipping and monotonic repair match linear ridge.

| Model | Local EMD | Paired-camera disagreement EMD |
| --- | ---: | ---: |
| Original RGB 100 mm 3-NN | 45.2408 | 33.6083 |
| Linear ridge RGB 100 mm, alpha 10 | 41.2059 | 28.8910 |
| Submitted grayscale multi-crop blend | 40.1178 | 31.5259 |
| **RBF kernel ridge RGB 100 mm, alpha 1** | **52.5486** | **25.4766** |

Lower is better. Local validation holds out each of the 24 physical soils,
excluding all its photos. The camera diagnostic covers the same 21 paired
soils. Kernel and linear ridge use different representations, so these fixed
alpha values do not imply matched effective regularization.

Kernel ridge improves five soils and worsens 19 compared with linear ridge;
it improves six and worsens 18 compared with the submitted blend. H374 worsens
by 81.39 EMD relative to linear ridge and by 57.35 relative to the blend.

**Decision: retain the submitted blend and make no new Kaggle submission.**
The fixed kernel candidate improves camera agreement but fails the local
accuracy criterion. Agreement between cameras alone is insufficient. This one
configuration underperforms; it does not establish that every kernel model would
fail. No additional bandwidths, penalties, or blends were tried after seeing the
result.

The next planned step is a frozen pretrained image encoder plus ridge, after
confirming the competition's pretrained-weight and external-data rules. The
earlier visual audit motivates richer features; this result alone cannot tell
us whether features or kernel settings limit performance.

## Reproduce and verify

After preparing data and running `make experiments` and `make multicrop`:

```bash
make kernel-ridge
make validate SUBMISSION=artifacts/submissions/kernel_ridge_rgb_100.csv
make test
```

The command writes OOF predictions, separate camera predictions, per-soil
comparisons, and a summary with input SHA-256 fingerprints under
`artifacts/experiments/kernel_ridge/`. The candidate has ten valid rows in
template order. All 49 earlier artifacts remain byte-for-byte unchanged.

All 46 tests pass. New checks include analytic two-soil shrinkage, an independent
augmented-system solution for the intercept, query-batch invariance, constant
inputs, curve repair, ID alignment, and preservation of input files. The
implementation uses NumPy and adds no dependencies.
