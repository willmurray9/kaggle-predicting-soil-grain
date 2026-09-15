# Physical texture spectrum and photo training — 15 September 2026

Both new directions improve local validation. Adding spectral texture reduces
EMD from **43.40571 to 40.53927**; training ridge on individual photos with equal
total weight per soil produces **42.28823**, while more than halving camera
disagreement. Both pass the declared two-metric submission rule and all four
earlier diagnostic screens. Spectral texture has lower EMD and is selected for
the one allowed upload. Kaggle completed that submission at **55.78511 public
EMD**, improving our previous best **61.11357** by **5.32846 (8.72%)**.
RGB + texture + spectrum is the new best submitted model.

Declaration: `41dd0a5e59b4f31634fecc2b5adbab5ac654295b`.
Producing code: `3bc7d28b0c6fda22628f6715a098facffd0aa4ee`.
See [the fixed plan](roadmap.md#physical-texture-spectrum-and-photo-training--declared-2026-09-15).

## Why these experiments

The September 15 preflight found us at **107/249 teams**, with best public EMD
**61.11357** versus the leader's **0.97357**. That confirms a large gap, but does
not establish competitors' methods. Data filenames/sizes and the seven official
competition pages remain unchanged. There were nine completed submissions,
zero today, and five daily slots available. [Leaderboard](https://www.kaggle.com/competitions/soil-grain-size-from-photos/leaderboard).

Most recent experiments changed the regressor or classification encoder. These
two instead test information discarded by our current pipeline: a broader
distribution of physical texture scales, and variation among photos of the same
soil. Both preserve ridge alpha 10, every label, the 100 mm crop, and whole-soil
validation. There is no parameter search or combination of the new ideas.

## Spectral texture

Convert the calibrated 256-pixel crop to the existing RGB-mean grayscale.
Subtract its mean under a separable Hann window, apply the window, and calculate
squared FFT2 magnitude. Sum radial power in six fixed image-wavelength bands:
**1–2, 2–4, 4–8, 8–16, 16–32, and 32–64 mm**. Frequency masks include `1/high`
and exclude `1/low` cycles/mm. Normalize by total power in those bands; total
power at most 1e-12 produces six zeros. Append these fractions to the existing
17 RGB/texture features, giving 23. Average photos equally in float64, then fit
ordinary ridge using training-only scaling.

Our four earlier squared-difference measurements sample spatial structure at
four offsets. The spectrum summarizes variation over a broader set of scales.
Fourier/wavelet approaches have been studied for visible sediment grain sizing,
which motivates these descriptors. Our fractions are **image texture powers**,
not measured grain diameters or mass fractions. Visible surface axes, occlusion,
shadows, and bulk laboratory mass distributions are different quantities.
[Primary wavelet study](https://onlinelibrary.wiley.com/doi/10.1111/sed.12049),
[USGS Fourier-method publication](https://pubs.usgs.gov/publication/70156415).

At 100 mm across 256 pixels, the sampling is 0.390625 mm/pixel. Clay and silt
grains are not directly resolved; those fractions still require inference from
visible properties. More texture bands cannot remove this physical limitation.

## Training on photos

Keep exactly the same 17 incumbent features, but give training photo j from
soil i weight `1 / number_of_photos_in_soil_i`. Each soil's total weight stays
one. Fit scaling from the training-soil feature means, retain the unpenalized
intercept, and solve weighted ridge with alpha 10 against repeated soil targets.
All photos of the held-out soil are excluded. Its pooled and separate-camera
means are queried in the same fit, before clipping and monotone repair.

For an affine prediction f, the squared photo loss for a soil decomposes into:

`average_j ||f(x_ij) - y_i||² = ||f(mean_j x_ij) - y_i||² + average_j ||f(x_ij) - f(mean_j x_ij)||²`.

Thus photo training adds a penalty for predictions changing between views of the
same soil. It uses the repeated views without treating them as independent new
labels. That variation can reflect real heterogeneity as well as cameras and
framing, so suppressing it is not always beneficial. Training still contains no
iPhone views. Because the model is affine before curve repair, querying a mean
feature vector equals averaging raw photo predictions; repairing each photo
first would be a different procedure.

## Local results

Lower is better. All 24 soils remain in fitting and primary validation.
Camera disagreement covers the same 21 pairs and is a sensitivity diagnostic.

| Model | Held-out EMD | Camera disagreement | Soils improved versus incumbent |
| --- | ---: | ---: | ---: |
| RGB + texture ridge | 43.40571 | 29.11306 | — |
| RGB + texture + spectrum | **40.53927** | 25.33273 | **15/24** |
| Ridge trained on weighted photos | 42.28823 | **13.91782** | 13/24 |

| Earlier diagnostic screen | Spectrum | Photo training |
| --- | ---: | ---: |
| Mean gain ≥ 1 EMD | +2.86643 | +1.11747 |
| At least 12/24 soils improve | 15 | 13 |
| Camera disagreement no worse | Pass | Pass |
| Mean gain without largest beneficiary > 0 | +2.27649 | +0.31120 |

The spectral gain is spread across several soils. F827 improves by **16.43508**,
H493 by **13.52186**, and H038 by **11.59692** EMD. F827 and H038 were difficult
cases with similar old features but very different target curves, so improving
both is useful evidence that the broader texture description adds information.
The largest regressions are H637 (−7.66824), H516 (−4.68327), and H617 (−2.84222).

Photo training helps H666 (+19.66164), H126 (+12.68666), and H038 (+11.54220),
but hurts H668 (−8.75650), H549 (−7.28076), and H405 (−6.39458). Its camera
improvement is much larger than its accuracy gain. The reduction is consistent
with the within-soil variation penalty, but does not prove transfer to iPhones.

## Kaggle result

Submission **56258597** completed at **55.78511 public EMD**, uploaded on
**2026-09-15 at 16:08:21.117 UTC**. The private score is unavailable. See the
[authenticated submissions page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/submissions)
and [submission log](submissions.md) for the record.

The local improvement transferred to this public evaluation. This is evidence
for the broader texture description, but it does not establish why top teams
score much better or guarantee improvement on the private leaderboard. No
settings changed after the public result, and the photo-trained candidate was
not uploaded. There are **ten lifetime submissions, one today, and four daily
slots remaining** under the API limit of five.

The receipt at
`artifacts/experiments/informative_submissions/2026-09-15_spectral_ridge.json`
was written before the request and finalized against the exact accepted
reference. It records the selected file hash, source/declaration commits,
validation decision, CI result, request times, and completed Kaggle response.

## Reproduction and verification

Run `make physical-photo`. It verifies the combined 17-feature cache and source
photo/calibration fingerprints, then reproduces the saved incumbent OOF, camera,
and test curves within **1.14e-13**. New artifacts are isolated under
`artifacts/experiments/physical_photo/`; the command never uploads automatically.

All **288 tests pass**. Tests include synthetic spectral frequencies,
affine-brightness and quarter-turn invariance, calibrated cropping, analytical weighted ridge,
equal soil influence, one-view equivalence, whole-soil exclusion, source and
reference corruption, and test-data independence. Independent implementation
review found no actionable issues; a synthetic weighted fit matched scikit-learn
within **2.13e-14**. Both [CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34992686018)
passed before upload. Independent output review confirmed valid curves, template
order, all 178 recorded source/output hashes, and a selected prediction file
numerically distinct from all nine prior uploads. All **151 prior artifacts
remain byte-identical**. No packages or external models were added.

| Candidate in `artifacts/submissions/` | SHA-256 |
| --- | --- |
| `rgb_texture_spectral_ridge.csv` | `7e34fa8cc8e8eaa455572ce93ca115077fdd085b41aad72a1745681d68212766` |
| `rgb_texture_photo_ridge.csv` | `8e45535b7ba1335a677752dbe6c721b26c7c4f247ca92ebc5c784676a5b055a3` |

The next larger representation experiment remains self-supervised DINO patch
features. A combination of spectral features and photo training is another
bounded follow-up, but neither has been evaluated in this batch.
