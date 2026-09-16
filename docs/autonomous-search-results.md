# Autonomous model search — September 16, 2026

The user authorized continued experiments and submissions until a public EMD below **55.78511**, or the five daily slots were exhausted. The [fixed batch](autonomous-search-plan.md) was declared before new scores in `e6d9f5ee5e6c7043ba5aa609ccd5e1fa66e682a0`.

**Outcome: all five daily submissions used; the public best remains 55.78511.** Seven candidates were evaluated locally.

## Local results

| Candidate | Whole-soil LOO EMD ↓ | Camera disagreement ↓ | Soils improved vs incumbent |
| --- | ---: | ---: | ---: |
| Spectral incumbent | 40.53927 | 25.33273 | — |
| dino_pls | 39.74220 | 12.15179 | 12/24 |
| spectral_dino_blend | 39.89860 | 20.63011 | 9/24 |
| spectral_mobilenet_blend | 39.90057 | 17.13367 | 11/24 |
| spectral_nested_ridge | 40.53927 | 25.33273 | 0/24 |
| dino_pca | 41.66782 | 22.10168 | 8/24 |
| mobilenet_pca | 41.82774 | 13.08667 | 10/24 |
| spectral_boost | 44.43476 | 42.41793 | 9/24 |

The frozen upload order is DINO PLS, spectral/DINO blend, spectral/MobileNet blend, DINO PCA/ridge, MobileNet PCA/ridge, then boosting. Nested spectral ridge selects alpha 10 in every outer fold and the final fit, reproducing the existing submission; it is excluded as a numerical duplicate.

PLS selects one supervised component in all 24 outer folds and the final fit. This replaces the old DINO model's unsupervised PCA with a direction learned jointly from training features and CDF targets. The local gains are modest: PLS improves 12/24 soils, and its mean gain becomes negative when the largest beneficiary is removed. The two blends also lose their mean advantage under that diagnostic. These are exploratory candidates, not robust evidence of a general improvement.

All new learned transforms and parameter choices are fitted inside the relevant training fold. Final full-training selection scores choose the submitted fit and are not additional validation estimates. Boosting uses the single fixed recipe; it is worse locally and less consistent across cameras. No weight search or post-public-result model changes are made.

## Public submissions

| Order | Candidate | Submission ref | Public EMD ↓ | Difference from 55.78511 |
| ---: | --- | --- | ---: | ---: |
| 1 | dino_pls | `56285554` | 69.70024 | +13.91513 |
| 2 | spectral_dino_blend | `56285568` | 60.65620 | +4.87109 |
| 3 | spectral_mobilenet_blend | `56285591` | 60.59547 | +4.81036 |
| 4 | dino_pca | `56285606` | 69.02875 | +13.24364 |
| 5 | mobilenet_pca | `56285626` | 67.06322 | +11.27811 |

All five are complete. **The daily allowance is exhausted; no public improvement was achieved.** The incumbent remains **55.78511** (submission `56258597`). Today's best is the spectral/MobileNet blend at **60.59547**, which is **4.81036 EMD / 8.62% worse**. Sixteen lifetime submissions are now complete. No further upload or experiment follows this stop condition. Boosting is sixth in the frozen queue and remains unsubmitted; nested ridge was a duplicate.

The three locally better candidates all perform worse publicly. In this batch, reduced disagreement between training cameras also fails to predict improvement on the unseen test cameras. This supports prioritizing a concrete investigation of validation/test mismatch in a future run, rather than assuming further small improvements on these same 24 holdouts will transfer. It does not identify the cause of the mismatch. No public-score weight tuning was performed.

## Reproducibility and verification

Producing code: `02ffc4cbd0a9aa38fad9713214c1fa54212adf1d`. Run with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 make autonomous-search
```

The first producing attempt (`533df3beaf0b2e01e95b5c910270f6ba4625aeea`) stopped on strict submission-schema validation: internal canonical labels such as `2.0` differed from the competition header `2`. A regression test reproduced it; exporting blends through the existing template helper fixed the spelling without changing any recipe or predictions. No upload occurred during the failed attempt.

The complete suite passes **510 tests**; the two pandas warnings arise from the existing cached-feature alignment helper and affect performance only. Independent reviews verified model/fold boundaries, exact blend arithmetic, source hashes, output alignment and the submission request safeguards. The spectral reference reconstructs within 1e-10. All 210 artifacts present before this search remain unchanged.

Inputs include 389 verified unique source records before adding producing code, the plan and prior-submission files. Pinned manifests connect the labels, photos, cached features and encoder checkpoints. Outputs and selection-table hashes are recorded in `artifacts/experiments/autonomous_search/summary.json`; receipts are saved before each network request under `artifacts/experiments/autonomous_search/receipts/`.

| Candidate CSV | SHA-256 |
| --- | --- |
| `artifacts/experiments/autonomous_search/dino_pca/dino_pca.csv` | `6d1be7b44e04117ceaec57b03b74e66284e2f82d0e0ee52ab675e3d9a4c28c64` |
| `artifacts/experiments/autonomous_search/spectral_dino_blend/spectral_dino_blend.csv` | `ae60ef3c5ce7a08f780cbdb128d46d1274a45208ba807141aa3c7654d1aec3fb` |
| `artifacts/experiments/autonomous_search/mobilenet_pca/mobilenet_pca.csv` | `273a3158ac3644fd85f5364cd90a88faed6d160956a59cd156220ef47866d40e` |
| `artifacts/experiments/autonomous_search/spectral_mobilenet_blend/spectral_mobilenet_blend.csv` | `2386b47548c0276c63ebb32aa7724ba55ba3b0458a9106c219a4f76180f32a98` |
| `artifacts/experiments/autonomous_search/dino_pls/dino_pls.csv` | `673d9623902733f5b0e0567bc27fcb44fd474925d3afe9eb9995b950e83b91fa` |
| `artifacts/experiments/autonomous_search/spectral_nested_ridge/spectral_nested_ridge.csv` | `ee408a3f278bc36f4f10dbd6d2098802aa68c5cb25d4e8665197c41161f16e7e` |
| `artifacts/experiments/autonomous_search/spectral_boost/spectral_boost.csv` | `5ff302deff53a8ee66d1af7c46b9d4733c04b12e653d81eff258d2986eed8d68` |

The supervised PLS audit independently used scikit-learn directly: final test curves and a complete held-out soil/camera prediction reproduce within 1.42e-14. The one-component choice is stable across folds, but the median per-soil gain is negative (−1.35202 EMD). Stable model selection is not evidence that the validation distribution matches the test distribution.

Both producing-code CI jobs passed: [run 35134763361](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35134763361).

Batch summary SHA-256: `7efbdbf20382e30a2d9ac3667b8f98ee5fc6da22dbb88196340ca1d26ba34989`.

The final artifact inventory in `artifacts/experiments/autonomous_search/verification.json` binds the 32 other batch files, including all five completed receipts. Its SHA-256 is `7cdd3c09e0659673f77f8f332e4a3cd7034ced2410b0d344b2bb0020a85c71fb`.
