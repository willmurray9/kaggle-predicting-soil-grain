# Frozen DINOv2 patch features — 15 September 2026

The fixed DINO model scores **41.66782 local EMD**, worse than spectral ridge's
**40.53927**. Camera disagreement improves to **22.10168**, but only 8/24 soils
improve. It fails the declared submission rule, so **no upload was made**.
Spectral ridge remains our best public model at **55.78511**.

Settings were fixed in declaration `70daa80002eedb906147f3a064acb0d41598fdd3`
before soil-feature extraction or new scores. Producing code:
`3e4a97a6bdeac02c1ac6f1481751f378510ecd17`.
See [the fixed plan](roadmap.md#frozen-dinov2-patch-features--declared-2026-09-15).

## Representation and fixed comparison

Use DINOv2 ViT-S/14 without registers. The official model was pretrained on
LVD-142M without class labels. Its patch features offer a different description
of visual structure from the handcrafted measurements and ImageNet classification
features already tested. This is a hypothesis about transfer to soil images,
not evidence that the encoder directly measures grain sizes or mass fractions.
[Official repository](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8),
[research paper](https://arxiv.org/abs/2304.07193).

Start with the existing calibrated 100 mm central crop rendered at 256 pixels.
Apply the official evaluation recipe: bicubic resize to 256, center crop to
224, tensor conversion and ImageNet normalization. The resulting field is
**87.5 mm**. Use the final block's **256 patch tokens**, after the model's
learned LayerNorm, and average them into **384 features**. Exclude the class
and register tokens; do not add unit-length normalization or concatenate layers.
Spatial averaging can discard useful information about grain arrangement.

Keep the encoder frozen in evaluation/inference mode, using CPU batches of eight.
Average photos equally per soil with float32 aggregation, as in our previous
frozen encoders. Fit eight-component PCA and ridge using the existing nested
whole-soil validation. Select alpha from **10, 100, 1000**, with larger alpha
winning exact ties. Scaling, PCA, and regression are fitted inside each inner
and outer training fold. Final full-training selection chooses the deployed
alpha; its score is not a validation estimate. All 24 soils remain in fitting
and primary validation, and the camera comparison covers the same 21 pairs.

The encoder never receives soil labels. This is one fixed candidate, with no
fine-tuning, alternate pooling, blend, resolution search, or fallback upload.
The submission rule requires strictly lower local EMD and camera disagreement
than spectral ridge. The four older screens remain diagnostics.

## Results and interpretation

Lower is better. All 24 soils and 21 camera pairs are retained.

| Model | Held-out EMD | Camera disagreement |
| --- | ---: | ---: |
| Spectral ridge reference | **40.53927** | 25.33273 |
| DINO patch mean + nested PCA/ridge | 41.66782 | **22.10168** |

| Diagnostic | Result |
| --- | ---: |
| Mean EMD improvement | −1.12855 |
| Soils improved | 8/24 |
| Mean improvement excluding largest beneficiary | −3.33431 |
| Earlier four-screen diagnostic | Fail; camera criterion alone passes |
| Declared strict improvement in both metrics | Fail |

Alpha **100** was selected in every outer fold and in the final full-training
selection. That consistency does not change the held-out score or justify
searching nearby settings against these outer results.

DINO helps H666 by **49.60396 EMD**, H126 by **41.84944**, and H366 by
**28.63301**. However, H405 worsens by **34.90733**, H668 by **21.62109**, and
G190 by **20.19976**; 16 soils worsen overall. Removing the largest beneficiary
from the diagnostic makes the average deficit larger. No soil is removed from
model training or primary validation.

The representation contains useful information for some soils, but this fixed
pipeline does not outperform the compact spectral features overall. Pretraining
may emphasize visual properties unrelated to bulk grain mass, and averaging
patch tokens or compressing to eight components may discard relevant detail.
The experiment does not isolate those possible causes. No different pooling,
PCA count, encoder, or blend was evaluated after seeing these scores.

The candidate is saved, not submitted. Today's count stays at **two of five**,
with **three slots remaining**. A possible next direction is a separately
declared comparison of an EMD-aligned training objective on the successful
spectral features; current ridge fits squared error. This is not yet evaluated.

## Provenance and reproduction

Upstream code: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
Source Python-file aggregate SHA-256:
`117c2d9278ba282500e5d3924dae634672bd19cf892f3f8f4800fbd80b7b6af4`.
Official checkpoint: `dinov2_vits14_pretrain.pth`, **88,283,115 bytes**.
Checkpoint SHA-256:
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
[Official checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth).

Run `make dino-pca` after the preceding spectral experiments, with the existing
vision dependencies installed. The code verifies prior source/photo hashes,
aligns spectral OOF/camera curves and recomputes reference metrics before
extracting DINO features. It pins the upstream revision, verifies cached source
and checkpoint digests, and loads the state dictionary strictly using
`weights_only=True`. Output includes source-file hashes, exact preprocessing,
pooling and runtime versions, features, held-out curves, alpha selections,
per-soil comparisons, and the candidate. The command never uploads automatically.

Fresh preflight at **16:40:47 UTC** confirmed **eleven completed lifetime
submissions, two of five daily slots used**, rank **83/249**, and unchanged
competition pages. The public, freely downloadable encoder meets the unchanged
external-model rule. [Competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules).

All **316 tests pass**, including 17 new encoder/driver cases. They cover the
fixed image recipe, patch-only averaging, frozen and batch-independent inference,
invalid/corrupted outputs and sources, reference checks, and exclusion of test
features from nested model selection. Independent code review found no
experiment-correctness issues. Both [CI jobs](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/34997449084)
passed on the producing code. Independent output review reproduced the metrics
and decision from saved curves, verified all prediction keys and source hashes,
and confirmed alpha selection and template order without refitting.

All **165 previous artifacts remain byte-identical**. The new record includes
**175 source/photo/feature/candidate hashes**, plus **156 upstream Python-file
hashes** and the checkpoint digest. No dependencies or existing predictors were
changed. The optional xFormers package is absent; the declared tensor batches
ran successfully using the upstream CPU implementation.

Candidate: `artifacts/submissions/dinov2_vits14_patch_pca8_nested_ridge.csv`.
SHA-256: `f9c04cd140e2fb26cd87db3843459110a5ad02ebc03172d141100f5656484003`.
Feature cache SHA-256:
`960ba6922944328eda4064e3e9f27f583c7fade50614514c960c4fdec75a42a2`.
Detailed outputs: `artifacts/experiments/dino_pca/`.
