# Official ResNet preprocessing — 14 September 2026

The official recipe improves PCA ridge from **45.02758 to 40.80177 local EMD**
and its fixed RGB + texture blend from **42.02843 to 40.25638**. Both pass all
four criteria against our incumbent. However, both increase camera disagreement
relative to their corresponding earlier versions, failing the additional
predeclared criterion. **Neither is submitted.** RGB + texture remains our best
public submission at **61.11357**; today's usage stays at one of five slots.

The [declaration](roadmap.md#official-resnet-preprocessing--declared-2026-09-14)
was committed as `7093b1c` before extracting features or scoring. Producing code
is `0cbb5537fdf3dbfca6e0df6b82a582e0dec1f5c4`. Settings and the submission screen
were kept fixed after seeing results.

## Controlled change

Apply `ResNet18_Weights.IMAGENET1K_V1.transforms()` to the existing calibrated
100 mm center crop, already resized to 256 × 256 with Lanczos. The recipe's
resize to a 256-pixel shorter side is a no-op here. Its central 224 × 224 crop
leaves a nominal **87.5 mm** field of view; ImageNet normalization stays the same.
The smaller field and changed encoder input size are tested together, so this
experiment cannot attribute gains to either individually. See the
[official recipe](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

The pretrained checkpoint stays frozen in CPU evaluation/inference mode. Average
512 float32 features equally across each soil's photos. Within every training
fit, standardize features, fit eight-component PCA, and fit ridge without
whitening or component rescaling. Select alpha from 10/100/1000 inside each
outer training fold; every photo of the held-out soil stays excluded. The
second candidate averages these predictions with RGB + texture at fixed 50/50
weights, aligning soil IDs and camera keys first.

## Results

Lower is better for both columns. Camera disagreement measures differences
between predictions from different cameras photographing the same held-out soil;
it is a sensitivity diagnostic, not an unseen-camera accuracy estimate.

| Model | Outer EMD, 24 soils | Camera disagreement, 21 pairs |
| --- | ---: | ---: |
| RGB + texture, current public best | 43.40571 | 29.11306 |
| Earlier PCA ridge | 45.02758 | 14.66981 |
| Official-preprocessing PCA ridge | **40.80177** | 16.90987 |
| Earlier fixed RGB + texture/PCA blend | 42.02843 | 17.90374 |
| Official-preprocessing fixed blend | **40.25638** | 19.41314 |

PCA improves 18 soils and worsens six against its earlier version. The blend
improves 17 and worsens seven. Against RGB + texture, PCA improves 15 soils and
the blend improves 17. The new blend is close to the historical grayscale crop
blend's 40.11782 local score, which transferred poorly to the public leaderboard.
Local ranking still needs an external check before claiming better test accuracy.

| Submission criterion | PCA | Blend |
| --- | ---: | ---: |
| Gain ≥ 1 EMD versus RGB + texture | +2.60394, pass | +3.14933, pass |
| At least 12/24 soils improve versus RGB + texture | 15, pass | 17, pass |
| Camera disagreement ≤ incumbent's 29.11306 | 16.90987, pass | 19.41314, pass |
| Positive gain without largest beneficiary | +0.99576, pass | +2.42556, pass |
| Gain ≥ 1 EMD versus matched earlier version | +4.22581, pass | +1.77206, pass |
| Camera disagreement ≤ matched earlier version | 14.66981 → 16.90987, **fail** | 17.90374 → 19.41314, **fail** |

For the blend versus RGB + texture, the largest gains are H126 (+19.79605),
H666 (+16.14241), and H366 (+12.88818). H516 worsens by 10.40760, H405 by 6.87316,
and H549 by 5.94374. The largest beneficiary contributes about 26% of the total
net gain, compared with about 67% for the previous blend. The new improvement
is more broadly distributed across these folds; all soils remain in fitting
and primary evaluation.

The camera criterion is a conservative budget rule, not proof these candidates
would lose publicly. The prediction files are retained for a future explicitly
planned comparison. Changing the rule after these scores would undermine this
round's predeclared decision. This batch makes no further upload or tuning pass.

## Regularization and interpretation

Outer folds select alpha **100 for 18 soils** and **10 for six**. The final fit
chooses **100**. Full-training selection EMD is 40.60538 at alpha 10, 40.58853 at
100, and 43.97237 at 1000. These scores choose the final penalty; they are not
additional validation estimates. The small gap between 10 and 100 cautions
against treating the selected penalty as strongly established.

The results show that preprocessing can materially change the usefulness of a
frozen representation on these soils. They do not establish whether smaller
spatial coverage, changed feature responses, or compatibility with pretraining
caused the gain. With only 24 labeled soils and many exploratory comparisons,
these remain candidate improvements rather than an unbiased performance claim.

## Reproduction and checks

Run `make official-preprocessing` with the optional vision dependencies and the
existing frozen, PCA, and physical-texture artifacts. It writes separate files
under `artifacts/experiments/official_preprocessing/`, including features, OOF and
camera curves, all alpha selections, per-soil comparisons, and a manifest with
`selected_submission: null`. The command does not upload to Kaggle.

- All **149 tests pass**, along with both [remote CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34894506339). Tests cover exact official input pixels, unchanged
  legacy output/metadata, frozen batch behavior, nested training isolation,
  key alignment, source/calibration changes, and the strengthened screen.
- Independent implementation review found no actionable issues; a separate output
  audit reproduced scores, submission decisions, and exact blend arithmetic. Both candidate
  CSVs validate against the ten-row template and contain valid cumulative curves.
- All **113 previous artifacts remain byte-identical**. The manifest fingerprints
  14 source files, 162 photos, the new feature cache, and both candidates.
  Source-photo, reference-input, and checkpoint checks preserve the comparison.
- No new dependencies, external training data, or pretrained checkpoint were
  needed. Competition rules remain unchanged from the recorded September 10 check.

| Unsubmitted candidate in `artifacts/submissions/` | SHA-256 |
| --- | --- |
| `official_resnet18_pca8_nested_ridge.csv` | `6b14b38b1452779876c089d359e35d08022661038e861017ef521580d12107a4` |
| `rgb_texture_official_pca8_blend.csv` | `ff75b009ae1025bd90264fd95a961ee4b5def5c62d3e2afde20d2f3cbcb2c4fb` |

Next, prioritize a bounded regressor comparison on the incumbent's compact
RGB + texture inputs. A second frozen encoder remains available, but the current
results also justify testing the prediction model without another representation
change. Declare that comparison and its evidence requirements before scoring.
