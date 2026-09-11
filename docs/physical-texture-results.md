# Physical texture and color control — 2026-09-11

Four contrast-normalized texture measurements did not improve local validation:
RGB ridge rises from **41.20591 to 43.40571 EMD**, and grayscale ridge rises from
**43.15793 to 45.34803**. Lower is better. Removing color improves agreement
between cameras, but worsens prediction accuracy on the held-out training soils.

The [fixed four-configuration comparison](roadmap.md#physical-texture-with-a-color-control--declared-2026-09-11)
was declared in commit `34a0bfb83340f5504fcf48a219d011b7565b3e3c`, before new
validation or public scores. The three new candidates remain the declared
submission batch irrespective of their local rankings. Public results and exact
upload provenance belong in the [submission log](submissions.md).

## Controlled comparison

Every configuration uses a center 100 mm crop rendered at 256 pixels, with the
same PPM correction and EXIF orientation. The original RGB input has 13 features;
its grayscale counterpart has seven. Each soil receives the equal mean of its
photos' feature vectors. Ridge alpha stays fixed at 10; standardization and
regression are fitted within each of the 24 whole-soil held-out folds. Separate
camera predictions exclude all photos of that held-out soil from training.

The four new features describe squared brightness changes across nominal
separations of 1, 2, 4, and 8 mm, divided by twice overall grayscale variance.
Average the horizontal and vertical means equally, using only valid pixel pairs.
At this resolution the separations round to 3, 5, 10, and 20 pixels, or 1.171875,
1.953125, 3.90625, and 7.8125 mm. Variance below 1e-12 returns zeros. Values can
exceed one and are not clipped. The normalization removes sensitivity to
unclipped affine brightness changes away from the constant-image guard.

These measure spatial texture, not particle diameters or grain counts. Some
separations overlap the original absolute-difference texture features; the
addition is squared, contrast-normalized variation. This experiment neither
changes the crop coverage nor tunes scales, regularization, or blend weights.

## Local results

| Inputs | Features | Held-out EMD | Camera disagreement (21 pairs) |
| --- | ---: | ---: | ---: |
| Original RGB reference | 13 | **41.20591** | 28.89103 |
| RGB + physical texture | 17 | 43.40571 | 29.11306 |
| Grayscale control | 7 | 43.15793 | 22.74678 |
| Grayscale + physical texture | 11 | 45.34803 | **22.32622** |

Adding texture worsens local error by **2.19980 EMD with RGB** and **2.19010 with
grayscale**. Ten soils improve and fourteen worsen in the RGB comparison;
eleven improve and thirteen worsen in grayscale. The closely matched aggregate
changes offer no evidence that the new texture descriptors solve a problem
specific to color features.

The largest texture regressions recur in both comparisons: H666 worsens by
16.68/16.43 EMD and H372 by 13.71/15.55 for RGB/grayscale respectively. H366
improves by 5.53/8.24. This is mixed behavior across soils, not one universal
improvement concealed by a single outlier.

Removing color without added texture worsens EMD by 1.95202 and improves only
7 of 24 soils, while lowering camera disagreement by 6.14424. H374 worsens by
16.16 and F827 by 8.47. The same 7/17 improved/worsened split occurs when removing
color with texture present. Camera stability remains a separate diagnostic;
it does not establish more accurate predictions.

With only 24 labeled soils, extra descriptors can add redundancy or unstable
relationships even when their definitions are physically interpretable. That
is a possible explanation, not proof that useful texture information is absent.
These four descriptors and this fixed ridge model are one bounded test.

## Reproduction and validation

Run `make physical-texture` after `make data` and `make experiments`. No new
packages or pretrained checkpoints are needed. Outputs under
`artifacts/experiments/physical_texture/` include the four new features for every
photo, all four OOF/camera predictions, paired per-soil comparisons, and a
manifest with settings and 172 input fingerprints. Existing RGB/grayscale
caches are aligned by complete split, soil, camera, and path keys.

- The original RGB reference reconstructs saved OOF predictions within
  4.27e-14, camera predictions within 5.69e-14, and test predictions within
  4.98e-14. The driver rejects differences above 1e-10 before writing candidates.
- All 109 tests pass, including analytic texture patterns, brightness/rotation
  invariance, invalid caches, whole-soil exclusion, equal photo weighting,
  submission alignment, and the reference reconstruction gate.
- Independent review found no actionable issues. All 93 earlier artifact files
  remain byte-identical. Each new candidate changes all ten test curves relative
  to RGB ridge; all candidates pass schema and cumulative-curve validation.

| Candidate file in `artifacts/submissions/` | SHA-256 |
| --- | --- |
| `ridge_rgb_texture_100.csv` | `f1a63c3591ac93d938f5ed64947be8cd481e62895c71febd5e365a004675911b` |
| `ridge_gray_100.csv` | `dc0bead104b134795a2c5c6011a283c4ae241c44fa38ed3d1e7afd59963c5d63` |
| `ridge_gray_texture_100.csv` | `f6e61138d27731d2641157c9e6dde81a85c89cc946eb7e4494f9be2faa522fad` |

The next roadmap step is controlled learned-feature preprocessing, such as
dimensionality reduction fitted separately inside each training fold. Keep this
texture batch fixed rather than searching new lags after these results.
