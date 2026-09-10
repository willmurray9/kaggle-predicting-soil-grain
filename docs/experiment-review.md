**Experiment review — 10 September 2026**

We have beaten the original image model. The first RGB/texture nearest-neighbor
baseline scored 45.24 local EMD; our submitted grayscale crop blend scores 40.12,
an 11.32% reduction. The 50 mm grayscale model has the lowest local error at
39.92, but considerably worse camera consistency. The current plateau is that
the subsequent camera-weighting, kernel, and frozen-image-feature experiments
have not displaced the submitted blend under our two-part selection rule.

This report covers completed experiments through code commit `f7fa0b2`. No new
model or proposed 50/50 blend was evaluated while preparing it. Numerical results
come from the repository's saved predictions and experiment reports.

**What we are predicting and how we measure progress.** Each soil has an
11-point cumulative grain-size curve: the percentage finer than each diameter
from 0.002 to 200 mm. Curves must be non-decreasing, stay between 0 and 100, and finish at
100. Our EMD implementation measures absolute differences between predicted and
measured cumulative percentages, weighted by intervals on the log-diameter axis.
Lower is better; 40 EMD is an error score, not 40% error or 60% accuracy.

There are 127 training photos but only **24 independently labeled physical
soils**. Multiple photos provide different views of the same target, not new
independent training examples. The test set contains 35 photos of ten soils.
Every local outer fold holds out one whole soil and all its photos, trains on
the other 23, and fits feature scaling on those training soils only. We average
the errors of all 24 held-out soils.

Our second measure compares predictions from separate camera views of the same
held-out soil. It covers 21 paired-camera soils. Lower disagreement means more
consistent predictions, but a constant predictor would achieve zero disagreement
while ignoring the images. Framing, lighting, arrangement, and photo selection
vary alongside the phone, so this is a camera-and-scene sensitivity diagnostic.

The initial audit matched all 162 readable photos to soils and cameras, found
no exact file duplicates, corrected physical cropping for image downsampling,
and aligned submission IDs. Exact-file checks do not establish that all scenes
or specimens are unrelated. Details are in [the baseline report](baseline-results.md).

**We started with image-blind references.**

| Reference | Local EMD | What it predicts |
| --- | ---: | --- |
| Equal mass in each grain bin | 102.02 | One fixed curve for every soil; scored on training labels |
| Mean curve | 89.89 | Average curve of the other 23 soils in each held-out fold |
| Median curve | 97.25 | Coordinate-wise median curve of the other 23 soils |

These tell us how far we can get without photographs. The original image model
reduced error by 49.67% versus the mean and beat it on 20 of 24 soils: substantial
evidence that the photos contain useful predictive information.

**The image experiments show a tradeoff between accuracy and consistency.**

| Experiment | Local EMD | Camera disagreement EMD | Result |
| --- | ---: | ---: | --- |
| Original RGB/texture, 100 mm, 3 neighbors | 45.24 | 33.61 | First image reference |
| Grayscale, 100 mm, 3 neighbors | 42.98 | 33.08 | Modest improvement |
| Contrast-normalized grayscale, 100 mm | 48.34 | 25.36 | More consistent, less accurate |
| Grayscale, 50 mm, 3 neighbors | **39.92** | 43.48 | Lowest local error; inconsistent camera views |
| Grayscale, 150 mm, 3 neighbors | 41.60 | 32.67 | Improves both measures over original |
| Linear ridge on original RGB/texture, alpha 10 | 41.21 | 28.89 | Improves both measures over original |
| **Equal blend of grayscale 50/100/150 mm predictions** | **40.12** | **31.53** | **Current submitted model** |
| Same crop blend, equal camera weighting | 42.42 | 38.34 | Rejected |
| RBF kernel ridge on original features | 52.55 | 25.48 | Rejected |
| Frozen ResNet-18 features + ridge, alpha 10 | 49.65 | 46.99 | Rejected |
| Frozen ResNet-18 + nested ridge selection | 49.92 | **21.19** | Rejected |

All image results use the same 24 outer held-out soils. These are local scores;
they are not eleven Kaggle submissions. The last row evaluates the entire
parameter-selection procedure using nested validation.

The original model takes a calibrated 100 mm center crop, resizes it to 256
pixels, and records 13 brightness, color, and texture measurements. It averages
features across a soil's photos, finds three similar training soils, and averages
their measured curves. It already incorporates physical scale and curve validity.

The first controlled batch changed color handling and crop size while retaining
three neighbors. Grayscale uses seven measurements instead of 13. Removing
color modestly helped, while normalizing each photo's contrast lost accuracy.
That is evidence against indiscriminately removing brightness information; it
does not tell us how much of that information represents soil versus acquisition
conditions. The smaller crop emphasizes local detail; the larger crop sees more
of a heterogeneous scene. Their different results motivated combining scales.

Linear ridge used the same original 13 features and learned a regularized linear
mapping to the curve. Its 41.21 looks good, but the gain is concentrated: H374's
error fell from 126.53 to 26.72. On the other 23 already-held-out predictions,
the original averages 41.71 and ridge 41.84. This is a diagnostic slice of saved
predictions, not retraining or an argument to remove H374. See
[the first-batch report](first-experiments.md).

The multi-crop model averages the three grayscale predicted curves, with fixed
one-third weights. We recompute EMD after averaging predictions; we do not average
their scores. It gives up only 0.20 EMD relative to the 50 mm crop while reducing
camera disagreement from 43.48 to 31.53. It met our predeclared requirement to
improve both measures over the original model. It improves 13 soils, ties one,
and worsens ten. This is why it was selected, rather than automatically taking
the lowest number in the local accuracy column.

Equal camera weighting tested an audit finding: some soils have more photos
from one phone than another. We first averaged each phone's features, then gave
the phones equal weight. Everything else remained fixed. All three component
crops became less accurate, and the blend worsened on both measures:

| Camera-balanced component | Local EMD | Camera disagreement EMD |
| --- | ---: | ---: |
| 50 mm grayscale | 41.62 | 46.79 |
| 100 mm grayscale | 43.93 | 52.12 |
| 150 mm grayscale | 45.77 | 36.01 |
| Equal prediction blend | 42.42 | 38.34 |

The balanced blend improves eight soils, ties six, and worsens ten. Helping F827
did not compensate for regressions elsewhere. The result rejects this weighting
choice, not the general importance of acquisition differences. See
[camera weighting](camera-balance-results.md).

Kernel ridge tested a nonlinear relationship using the original features, one
fixed RBF bandwidth (`gamma = 1/13`), and penalty `alpha = 1`. It scores 52.55,
worsening 19 soils relative to linear ridge. Its better camera agreement did not
translate to accuracy. One bandwidth and penalty do not rule out kernel methods;
they rule out this configuration as our next submission. See
[kernel ridge](kernel-ridge-results.md).

Frozen ResNet-18 replaced the hand-written measurements with 512 image features.
We used public ImageNet weights after checking competition rules, kept the encoder
and its normalization statistics frozen, and trained only ridge. There was no
neural-network fine-tuning. This fixed model scores 49.65 and improves only six
soils versus the submitted blend. It makes a striking improvement on F827, from
the blend's 102.89 error to 13.21, but a striking regression on G190, from 39.00
to 145.38. Richer features help some cases without being better overall.

Our encoder preprocessing preserved the full calibrated 256-pixel physical crop;
it did not apply the official recipe's additional 224-pixel center crop. That was
declared in advance to preserve coverage. The [official weight recipe](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
provides a useful untried preprocessing comparison. We cannot attribute the poor
result to this difference without testing it. See
[frozen features](frozen-resnet-results.md).

Nested ridge then asked whether choosing a stronger penalty could fix that
model. Within each outer fold, it selected among 10, 100, and 1000 using only
the remaining 23 soils; each inner model trained on 22. The chosen penalty was
1000 for 16 outer folds and 100 for eight. Accuracy remained worse at 49.92,
although camera disagreement fell to 21.19.

For the final test model, selection on all 24 training soils produced scores
49.65, 46.10, and 45.46 for penalties 10, 100, and 1000 respectively. **45.46 is
the score used to select the final penalty; 49.92 is the nested validation
result.** Reporting the minimum tuning score as an independent evaluation would
be optimistic. This distinction follows the rationale for
[nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Repeatedly choosing model families using these same 24 outer soils still makes
the overall research process exploratory. See [nested ridge](nested-ridge-results.md).

**Only the multi-crop model has been submitted to Kaggle.** Submission `56131658`
completed with public EMD **76.75797**, versus its local 40.1178. Consequently,
we do not know the public ranking of the other candidates. Training photos use
Motorola and Samsung phones; test photos use iPhone 14 and 16. The soils and
evaluation population also differ. The score gap is real, but one submission
cannot tell us how much comes from camera differences, soil composition, or
optimistic local model selection. The [submission log](submissions.md) records
the exact file hash and code commit.

**My interpretation is that the simple model matches several constraints of
this problem unusually well, while our evidence for alternatives remains small.**
The following explanations have different levels of support:

1. **Few independent soils constrain what we can learn.** We have 23 training
   rows per outer fold, whether each row contains seven features or 512. More
   photos, crops, or augmentations do not create new independently labeled soils.
   An unstable relationship between many features and a small set of curves is
   plausible. Strong penalties being selected is consistent with this concern,
   but we have not established a training-versus-validation gap that proves
   overfitting is the main cause.

2. **The baseline contains useful assumptions.** Physical crop calibration makes
   texture scales comparable. Neighbor averaging borrows complete measured curves
   and automatically preserves their constraints. The crop blend combines scales
   without learning weights. My hypothesis is that these restrictions help with
   sparse data. ImageNet features were learned for a different task; their added
   information need not emphasize grain size. We have tested only one such encoder.

3. **Our images and summaries can lose relevant information.** A rendered 100 mm
   crop at 256 pixels represents about 0.39 mm per pixel, much larger than several
   fine-grain support diameters. Those individual fine grains cannot be resolved
   directly in that rendered crop; their fraction must be inferred indirectly.
   Center crops also omit spatial variation. The F827/H038 audit found close
   feature neighbors whose measured curves differ by 191.47 EMD. This supports
   investigating representation and coverage. It does not establish an irreducible
   error floor or incorrect labels.

4. **A few difficult soils move the average substantially.** The submitted blend's
   improvements on H374 and H405 sum to 124.51 EMD across those two soils, slightly
   more than its net improvement of 122.95 across all 24. Across the remaining
   22 saved predictions, the combined change is slightly negative. This is not
   broad, uniform improvement. Likewise, frozen and nested ridge each improve
   only six soils versus the blend. These diagnostics help explain fragile rankings;
   they are not grounds for excluding difficult examples.

5. **Camera consistency is useful but insufficient.** Contrast normalization,
   kernel ridge, and stronger frozen-feature regularization all improved camera
   consistency while failing to improve the submitted model's accuracy. Our
   requirement to improve both metrics is deliberately conservative. It reduces
   one risk, but it cannot guarantee a better public score on unseen phones.

6. **We have not exhausted models, representations, or objectives.** Kernel ridge
   had one fixed setting. The nearest-neighbor count is still three. Frozen
   features had one encoder and physical crop. Ridge minimizes squared target
   error plus a coefficient penalty, whereas our competition score uses weighted
   absolute curve differences. That objective mismatch is a testable possibility,
   not an established explanation for the results. The
   [ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
   makes this distinction explicit.

There is also a known limit in the nearest-neighbor design: it cannot extrapolate
beyond its neighbors' curves. F827 is finer at 0.063 mm than every other training
soil, so an average of other soils cannot reach its measured value when it is
held out. A simple model being our current reference does not make it adequate
for every soil. The [error audit](error-audit.md) provides the evidence.

**I would continue in this order, with each comparison declared before scoring.**

| Priority | Experiment or diagnostic | What it would answer |
| --- | --- | --- |
| 1 | Fixed 50/50 blend of submitted multi-crop and nested ridge | Can their different errors offset each other? Reuse aligned held-out soil/camera predictions; no weight search. This is the already documented next experiment. |
| 2 | Small nested search over neighbor count, such as 1/3/5/7, keeping crops fixed | Have we left easy gains in the strongest model family unexplored? Consider distance weighting separately. |
| 3 | Fixed spatial sampling across each photo, with the same simple features | Does broader coverage help soils whose large grains enter and leave the center crop? Keep all patches from a soil in the same fold. |
| 4 | A small set of physical texture measurements at several scales | Can more useful image information help without adding hundreds of features? Frequency/texture summaries or coarse-particle cues are candidates, not established measurements of the full grain distribution. |
| 5 | Compact learned features or a controlled frozen-encoder comparison | Compare fold-fitted dimensionality reduction or a weight-recipe preprocessing change, one at a time; consider a second encoder afterward. |
| 6 | Small alternative regressors and an objective closer to EMD | Try tightly bounded kernel tuning, support-vector regression, or a shallow tree ensemble on compact inputs. An absolute-error curve model is another distinct comparison. |
| 7 | Limited encoder fine-tuning | Adapt a small part of a pretrained model only after the cheaper experiments provide stronger evidence. With 24 soils this has substantial model-selection risk. |

I would prioritize spatial coverage and stronger simple baselines over a broad
catalog of large models. Fine-tuning and training from scratch are very different
commitments; the current data provides little reason to train a large network
from scratch.

Alongside those experiments, strengthen the evaluation. Use soil-grouped split
sensitivity checks, inspect errors by soil and grain-size interval, and run
directional camera-transfer diagnostics while excluding every image of the
evaluation soil from training. These checks cannot manufacture independent data
or fully simulate the unseen iPhones. Independently verified soil provenance,
additional distinct labeled soils, or representative calibration data would be
valuable if legitimately available under the competition rules; none is currently
in hand. Do not infer label errors from model residuals.

Keep the public leaderboard as a limited external check. The current promotion
rule requires improvements over 40.1178 local EMD and 31.5259 camera disagreement,
plus changed, valid test predictions. Any revision to that rule should be justified
before looking at the next candidate's results. Keep small experiment batches,
record failed comparisons, and avoid extensive tuning against either 24 soils
or the public leaderboard.

The implementation already has 63 passing tests at `f7fa0b2`, including grouped
validation, inner-fold isolation, ID alignment, valid curves, frozen-encoder
behavior, artifact preservation, and numerical ridge checks; both CI jobs passed
for that commit. The remaining uncertainty is about modeling and generalization,
with no known unresolved pipeline defect from the completed reviews.
