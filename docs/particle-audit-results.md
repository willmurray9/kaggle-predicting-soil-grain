# Coarse-particle feasibility audit — 16 September 2026

**Reject this detector before predictive modeling.** The fixed watershed
method sometimes follows isolated pebbles, but repeatedly merges fine material
into large regions and splits individual coarse grains. Both independent visual
reviews reached the same decision across all eight photos. No model, CDF
candidate or Kaggle submission was produced. Spectral ridge remains best at
**55.78511 public EMD**; all five daily submission slots remain available.

Declaration: `c2f128b960a5d8acadfecacfa2882cf45a8c1b95`.
Producing code: `d33961919cd92fe045b69203a979a86c0788e195`.
See [the fixed protocol](roadmap.md#coarse-particle-feasibility-audit--declared-2026-09-16).

## Method and panel

The question was whether individual visible regions could support particle
counts and projected-size measurements. Earlier granulometry measured removed
image contrast; it did not identify individual regions. This audit tests that
missing step before deriving another set of model features.

Select positions 0, 7, 14 and 20 in the sorted list of 21 training soils with
both Motorola Edge and Samsung A52 views: **F827, H183, H516 and H668**.
Use the lexicographically first photo for each camera. Selection preceded
viewing and used no soil targets or model errors. These are different views of
the same soils, not registered photographs of the same individual particles.

Retain the central calibrated 100 mm crop without resizing: **460 pixels at
4.5968 pixels/mm** for Motorola and **455 at 4.55252 pixels/mm** for Samsung.
The files already have reduced resolution; preserving these pixels does not
recover the cameras' original detail. At these scales, 2 mm spans about nine
pixels, while 0.063 mm spans less than one pixel.

Compute RGB-mean grayscale Gaussian-gradient magnitude with sigma
`max(0.5 pixel, 0.25 mm × PPM)`. Choose the lowest gradient quartile, open it
once with a 0.5 mm-radius disk, and label eight-connected components as seeds.
Run one eight-connected watershed on the uint16-scaled gradient, with positive
seeds only. Flat images or an empty seed set return no regions. The algorithm
partitions the entire image, including matrix and shadows; it has no grain
versus background classifier. [SciPy watershed](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.watershed_ift.html),
[Gaussian gradient](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html).

Record all regions, including small and crop-truncated ones. Equivalent-circle
diameter is `2*sqrt(pixel_count / (PPM²*pi))`. Interior regions at least 2 mm
are highlighted in cyan, truncated boundaries in magenta, smaller regions in
gray and actual seed pixels in orange. IDs identify the six largest interior
candidates in each overlay. These diameters are projected-region measurements,
not sieve diameters or laboratory mass fractions.

## What the overlays show

| Soil | Motorola regions / interior ≥2 mm | Samsung regions / interior ≥2 mm | Visual finding |
| --- | ---: | ---: | --- |
| F827 | 28 / 12 | 45 / 28 | Broad fine-matrix patches become false coarse candidates in both views. |
| H183 | 173 / 132 | 165 / 120 | A large stone is split internally; some isolated stones are captured, but mixed regions persist. |
| H516 | 184 / 141 | 183 / 138 | Some pebble outlines look plausible; large regions also merge stones with intervening matrix. |
| H668 | 88 / 56 | 78 / 52 | Large irregular regions contain many visible particles in both views. |

The counts describe algorithm output, not observed grain counts or segmentation
accuracy. Of **944 total regions**, 679 are interior candidates at least 2 mm,
229 touch the crop boundary and 36 are smaller interior regions.

The clearest counterexample is **F827, Motorola region 12**. Its 30,070 pixels
give an area of **1,423.06 mm²** and equivalent diameter of **42.57 mm**, yet
the source image shows a broad patch containing many small particles. An
area-weighted grain-size distribution would give this false region substantial
influence. F827 Samsung regions 37/40 fail similarly.

On H183 Motorola, the conspicuous stone near **(35 mm, 28 mm)** has multiple
internal cyan divisions. H183 Samsung region 80 follows a prominent stone
reasonably, but regions 31/34/50 include mixed material. H516 regions
87/103/116 in Motorola and 105/138 in Samsung merge particles and matrix.
H668 regions 45/51/65 and 25/39/43 show the same false-region problem.

This meets the declared stopping rule. No seed threshold, smoothing scale,
opening radius or alternate algorithm was tried after viewing the overlays.
The audit rejects this heuristic, not all particle segmentation. Another
approach would first need credible boundaries on a fixed panel; manually
annotated outlines would be needed to quantify segmentation accuracy.

## Reproduction and checks

Run `make particle-audit`. Outputs in `artifacts/experiments/particle_audit/`
include eight source-resolution crops, eight seed/region array files, four
paired overlays, complete photo/region tables and a manifest. `review.json`
records the subsequent visual decision and binds it to the manifest digest.
The original manifest retains its pre-review status; the review file is the
final decision. The command does not load soil target values or test images.

All **452 tests pass**, including 39 new detector/driver checks. Tests cover
EXIF and calibration, retained source pixels, seeds/connectivity, flat/no-seed
cases, positive-seed preservation, calibrated area and diameter, truncation,
source corruption and output provenance. Both
[CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35132050057)
passed on the producing commit. Independent artifact review reconciled crop
coordinates, arrays and measurements, verified all **34 recorded input/output
hashes**, and confirmed all **186 previous artifacts remain byte-identical**.

Manifest SHA-256:
`be4f8807e95165444d1d192f1ee5836597f230712494b095cc7cbbf030a0e0ab`.
Review SHA-256:
`4dde947d1256e3d2adc6023d6033fcd610bb3d792ff0de81e43f748ec2b7f931`.
Code and findings are tracked in Git; photo-derived artifacts remain local.

The September 16 preflight found **eleven completed lifetime submissions, zero
today**, rank **90/259**, and unchanged 165 published files and seven official
competition pages. The best score remains **55.78511** and the deadline remains
November 30, 2026 at 11:00 UTC. This audit uses no submission capacity.
