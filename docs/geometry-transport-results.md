# Image granulometry and distribution transport — 15 September 2026

Neither new direction beats spectral ridge. Geometry-only ridge scores
**48.33618 local EMD**, and distribution transport scores **47.47930**, versus
the reference's **40.53927**. Both increase camera disagreement, so neither
qualifies for submission. Spectral ridge remains best publicly at **55.78511**.

Declaration: `f51647f4cd100d20c81d44f8e76b226ff568ce84`.
Producing code: `6b14133aad80fdd6b4ad797a32e89cbc7ac96377`.
The fixed numerical tolerance described below was added with the producing code,
from a synthetic boundary test before any new soil scores.
See [the protocol](roadmap.md#image-granulometry-and-distribution-transport--declared-2026-09-15).

## Two different hypotheses

**Image granulometry changes what we measure.** Start from the calibrated
100 mm grayscale crop rendered at 256 pixels. Apply square morphological
openings at widths 3, 5, 11, 21, 41 and 83 pixels, independently to the same
image. Opening removes bright structures that cannot contain the square;
repeat on the inverted image to measure dark structures. Record successive
losses of image intensity, normalized separately for each polarity. These
twelve measurements are the entire input to ridge; no color or spectral
features are appended. [SciPy opening definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_opening.html).

The aperture widths represent approximately **1.17–32.42 mm**. They describe
removed image contrast, not grain counts or laboratory mass fractions. A fixed
83-pixel reflected border supplies context while every measurement uses the
original central region. Reflection can enlarge edge fragments; shadows,
overlap and unresolved fines remain limitations. The fixed square-size tests
verify geometry that an intensity histogram alone cannot distinguish.

**Distribution transport changes what we predict.** Keep exactly the reference's
23 features, but transform each training CDF into 1,000 log-diameter quantiles.
Fit ridge to those quantiles, apply bounded isotonic regression to restore their
ordering, then count mass below each of the eleven required diameters. For
example, the log-size midpoint between point masses at 0.002 and 0.2 mm is a
point mass at 0.02 mm; averaging their CDF heights produces a two-size mixture.
This tests a different assumption about how soil distributions vary.
[Quantile projection in distribution regression, §6.1](https://anson.ucdavis.edu/~mueller/frechet26.pdf).

The 1,000 outputs are a numerical representation of the same labels, not
additional independent observations. Squared quantile loss relates to W₂²;
the actual evaluation remains the competition's W₁/EMD. CDF reconstruction
uses a fixed 1e-12 log-diameter equality tolerance so floating-point roundoff
does not move an exact support atom into the next bin. Label conversion and
inversion alone cost **0.10229 mean EMD**, with maximum threshold error
**0.05 percentage point**. That representation error is small compared with
the observed model deficit, but this diagnostic does not bound its effect
after fitting and threshold inversion.

Both models retain alpha 10, an unpenalized intercept, equal float64 photo
means and all 24 training soils. Scaling and fitting use only the training
soils of each leave-one-soil-out fold. Camera evaluation uses the same 21
pairs. Neither experiment includes tuning, a blend, or a combined model.

## Results

Lower is better. Improvements and soil counts compare against spectral ridge.

| Model | Held-out EMD | Camera disagreement | Soils improved |
| --- | ---: | ---: | ---: |
| Spectral ridge reference | **40.53927** | **25.33273** | — |
| Twelve granulometry features + ridge | 48.33618 | 35.15894 | 7/24 |
| Log-quantile transport ridge | 47.47930 | 31.54698 | 10/24 |

| Diagnostic | Granulometry | Transport |
| --- | ---: | ---: |
| Mean EMD improvement | −7.79690 | −6.94003 |
| Improvement excluding largest beneficiary | −9.22051 | −9.05437 |
| Earlier four-screen diagnostic | Fail | Fail |
| Declared strict gain in both metrics | Fail | Fail |

Granulometry helps H126 by **24.94599 EMD**, H038 by **15.58146** and H037
by **13.62074**, but hurts H374 by **84.68171**, F827 by **23.50273** and
H637 by **23.24481**. H374 is the only soil photographed with the Motorola
Edge 60 Fusion. Soil identity and camera are therefore confounded in that
case; its large error cannot establish a camera cause. Seventeen soils worsen
overall, so this is not solely a single-soil failure.

Transport helps H516 by **41.68994 EMD**, H616 by **30.98216** and H038 by
**27.25358**, but hurts G190 by **40.23086**, H372 by **31.00977** and H181
by **27.48028**. With identical image inputs, changing the target geometry
substantially redistributes errors without improving their average. Possible
causes include the W₂/W₁ objective mismatch and mass crossing discrete sieve
thresholds during inversion. This experiment does not isolate either cause.

Saved held-out predictions show a systematic shift toward coarser distributions:
transport underpredicts passing percentage at every nonfinal threshold on
average. Bias is **−13.06 percentage points at 0.2 mm** and **−11.27 at
0.063 mm**, versus **−0.18 / +0.36** for spectral ridge. The 0.2, 20 and
63 mm thresholds contribute **5.26 of the 6.94 extra EMD**. These diagnostics
describe the saved predictions; no bias correction was fitted afterward.

These results reject the two fixed pipelines as replacements for our current
model. They do not prove that all morphological or distribution-valued methods
are unsuitable. No aperture, quantile-count, regularization or blend search
followed the scores.

The next proposed direction is to represent **mixtures within a photo** through
the distribution of local patch measurements. Earlier spatial coverage averaged
more crops; a compact measure of patch variation could distinguish a uniform
texture from a fine/coarse mixture with a similar mean. It would still be an
image proxy, with all patches grouped by soil. This hypothesis is not evaluated
here and requires its own fixed comparison before any new scores.

## Submission, provenance and reproduction

The declared rule allows at most one upload, requiring strictly lower local
EMD and camera disagreement. Neither qualifies; **no upload was requested**.
There are eleven completed lifetime submissions, two of five used on September
15, leaving three slots, confirmed from fresh Kaggle history at **17:16:03 UTC**.
See [the submission log](submissions.md).

Run `make geometry-transport` after the existing spectral experiments. It
checks recorded source/photo hashes and reconstructs the reference's held-out,
camera and submission curves before extracting any new features. Maximum
reference difference is **5.69e-14 percentage point**. Outputs include the
162×12 feature cache, both sets of 24 held-out and 45 camera predictions,
per-soil comparisons, two valid ten-soil candidates and a source manifest in
`artifacts/experiments/geometry_transport/`. Existing dependencies suffice.

All **384 tests pass**, including 68 new cases covering known shapes and
physical calibration, reflected boundaries, brightness/polarity behavior,
quantile round trips, true isotonic projection, training-only fitting, test
independence, source corruption and submission selection. Independent reviews
found no correctness issues in either model or the experiment driver.
Both [CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34999832224)
passed on the producing commit. An independent artifact audit recomputed the
metrics and decision from saved curves, checked prediction keys and template
order, verified all **178 source/photo/feature/candidate hashes**, and confirmed
all **173 prior artifacts remain byte-identical**. No refitting was needed.

| Saved candidate | SHA-256 |
| --- | --- |
| `artifacts/submissions/granulometry_ridge.csv` | `b6d795a8c923d44cd521a77adc6f4f1f305a321682afbff29601af4005f2347d` |
| `artifacts/submissions/transport_ridge.csv` | `8516dda7e7b6bb8967190b515ddbb752509be78805aa135230a532be78b8e595` |

Feature-cache SHA-256:
`da128c36412fee1a760def8b9ce9f1ff9aec35322aacf394dac6df5d8b8012e3`.
Code, tests and this report are tracked in Git; generated artifacts remain local
under the repository's existing policy.
