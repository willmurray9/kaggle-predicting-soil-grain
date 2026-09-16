# Strategy review — September 16, 2026

**Recommendation:** pause model swapping, measure transfer between cameras, then test one compact distribution model and one genuinely different visual baseline. The current public best remains **55.78511**. Today's five submissions are exhausted; this review runs no new experiments or uploads. The [September 17 plan](superpowers/plans/2026-09-17-domain-and-representation.md) is for a later user-requested session.

## What the leaderboard gap actually means

The official leaderboard snapshot on September 16 places us **90th of 259 teams**. The leader scores **0.92742**, about **60.15 times lower error**. Two teams are below 1, four below 5, and seven below 10. There is more than one unusually low result. These scores do not reveal the teams' methods or their private performance. [Leaderboard](https://www.kaggle.com/competitions/soil-grain-size-from-photos/leaderboard).

The metric weights sum to `log10(200 / 0.002) = 5`. Dividing EMD by five expresses the weighted average vertical error in the cumulative curve: ours is **11.157 percentage points**, the leader's **0.1855 percentage points** on the evaluated public cases. This is an error ratio, not a claim that their model is 60 times more accurate in every sense.

A June 1 competition reply states: “The public leaderboard always uses the same 3 soils out of the 10 test soils.” The official Evaluation separately specifies a final score of **30% public + 70% private**. The weights alone do not establish the sample counts; the discussion supplies that information. Three public cases can strongly favor a model that happens to fit those particular soils. We do not know which IDs are public. [Public-split discussion](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/702872#3465152), [Evaluation](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview/evaluation).

The **“possible Private Test-Set Leak?”** discussion asks essentially our question. A September 3 reply reports realistic private scores and hypothesizes public-split overfitting. Its author identity is absent from the retrieved API response, and it predates today's leading submission. This is a reported explanation, not proof about the leaders or evidence of misconduct. [Discussion](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/737406#3520559).

**Assessment:** luck from the small public sample could contribute, and repeated public selection could amplify it. We cannot quantify either explanation from the available evidence. We should neither dismiss the leaders as lucky nor treat their public scores as demonstrated near-perfect generalization. There are still substantial gaps in our own approach.

## What we have tried

These are representative matched local/public results, with lower EMD better. Local means leave one entire soil out, not random photo splitting. Full details remain in the linked historical reports and [submission log](submissions.md).

| Approach | Local EMD | Public EMD | What it taught us |
| --- | ---: | ---: | --- |
| Image-blind training mean | 89.89 | Not submitted | Images contain useful predictive signal. |
| Grayscale 3-NN, equal 50/100/150 mm blend | 40.12 | 76.75797 | Strong local similarity does not guarantee test transfer. |
| Original 13-feature RGB ridge | 41.21 | 61.87967 | Simple learned regression transfers better than the crop blend. |
| 17-feature texture ridge | 43.41 | 61.11357 | Worse local EMD can accompany better public EMD. |
| **23-feature spectral ridge** | **40.54** | **55.78511** | Our best public model; broad texture information helped. |
| Spectral ridge with weighted photo training | 38.86 | 59.84769 | Our best local score did not become our best submission. |
| Frozen ResNet, nested ridge | 49.92 | 63.01764 | Generic pretrained features alone were insufficient. |
| Frozen DINO, PCA/ridge | 41.67 | 69.02875 | A stronger encoder is not automatically a better soil representation. |
| Frozen MobileNet, PCA/ridge | 41.83 | 67.06322 | A smaller encoder also failed to transfer. |
| DINO, supervised PLS | 39.74 | 69.70024 | A modest local improvement was fragile across soils. |
| Spectral/DINO fixed 50/50 blend | 39.90 | 60.65620 | Complementary local errors did not produce a public win. |
| Spectral/MobileNet fixed 50/50 blend | 39.90 | 60.59547 | Same pattern with another encoder. |

We also tested nested neighbor counts and ridge penalties, normalization, camera weighting, multiple crop locations, official encoder preprocessing, kernels, SVR, shallow trees, boosting, patch variability, granulometry, and a quantile-based distribution target. Simple watershed segmentation failed an eight-image feasibility check before regression. This does not exhaust modern segmentation, but it gives no reason to keep tuning that implementation. See the [roadmap](roadmap.md), [geometry/transport report](geometry-transport-results.md), [particle audit](particle-audit-results.md), and [latest search](autonomous-search-results.md).

Across our 16 completed submissions, the descriptive local/public Spearman correlation is only **0.39**; Pearson is **0.13**. Camera disagreement/public Spearman correlation is **0.17**. These are selected, correlated experiments evaluated on a tiny public sample, not independent observations or a statistical verdict against validation. They do show that our current ranking signals have been unreliable.

## Why we appear to have plateaued

1. **We changed regressors more than the information they receive.** Most models compress a physically calibrated, resized image into averaged descriptors or embeddings. We have tried spatial coverage, but its simple averaging can still erase distributions of particle sizes and local variation. Frozen encoders are not soil-specific training; we have not fine-tuned a model.
2. **Our validation tests unseen soils, but generally familiar cameras.** The test photos use iPhone 14/16; most training images use other phones. The existing camera diagnostic measures disagreement between views of the same held-out soil. It does not train without a camera and then measure accuracy on a new soil from that camera. Even a useless constant predictor has zero disagreement. Training-camera robustness may not transfer to iPhones or different geological sources.
3. **The fine end accounts for most error.** In the incumbent's saved held-out predictions, **55.9%** of total weighted error occurs at CDF supports at or below **0.2 mm**, and **71.4%** at or below **0.63 mm**. These are cumulative-threshold error contributions, not independent mass-bin errors. Counting large visible stones alone misses much of the scoring problem.
4. **There is a real information limit, plus avoidable compression.** For 124 of 127 training photos, downloaded resolution is about **4.55–4.60 pixels/mm**: a 0.2 mm grain is under one source pixel. Our common 100 mm crop is then reduced to 256 pixels, making that grain roughly half a model pixel. Test iPhones supply about 13.94/19.53 pixels/mm before resizing. Higher-resolution processing might preserve useful texture, but cannot recover fines that the training photos never resolved.
5. **Twenty-four labels are still twenty-four labels.** The incumbent's per-soil EMD has mean 40.54, median 34.66, and standard deviation about 28.46. H038, H616 and H126 contribute 27.75% of total error. The three latest locally winning candidates all lose their mean advantage when their single largest beneficiary is excluded. This is a sensitivity check, not permission to drop difficult soils.
6. **Nested tuning is necessary but insufficient.** It protects parameter selection within an experiment. Repeatedly choosing new representations using the same 24 outer holdouts can still overfit our research decisions. We have no untouched local confirmation set; tomorrow's additional checks will also be exploratory.

The review found no evidence of a current metric or CSV-order bug. The corrected H374 data are already installed. File hashes establish byte identity, not that differently named soils cannot share a source or near-duplicate scene; a bounded cross-ID provenance check remains useful if a concrete concern emerges. [H374 correction](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/702039#3500091).

Our engineering has been careful, but we should have investigated the competition-specific evaluation and domain literature earlier. More passing tests establish reproducibility; they do not establish that the validation population matches the leaderboard.

## What outside research adds

**A soil-specific representation prior.** Organizer Enrico Soranzo's GRAI3 paper uses **58 soils / 321 images**, physical scale normalization, a pretrained EfficientNet, and a **two-parameter Weibull curve**. Its reported parameter R²/MAPE are not this competition's EMD, so its results are not directly comparable. The cheap transferable idea is to test whether two curve parameters capture our labels before training a specialized image network. [GRAI3 paper](https://doi.org/10.1016/j.compgeo.2026.108362).

**Useful public ideas, with source-level caveats.** We inspected four notebooks without executing them or downloading their model weights. None supplied a verified public score in the retrieved source/output:

| Public source | Idea worth considering | Why not copy its results blindly |
| --- | --- | --- |
| [ConvNeXt + Weibull](https://www.kaggle.com/code/ambrosm/soil-with-convnext-and-weibull) | Physical crops and compact curve output | Old 25-soil data; approximate metric weights; saved OOF uses the final epoch while test inference loads a selected checkpoint. |
| [DINO + square-root masses](https://www.kaggle.com/code/nomannic19/cv-psgsdi-dinov2-features-sqrt-mass-ridge) | Predict nonnegative mass fractions; retain patch mean and variability | Large nonnested representation/parameter search; the selected CV score is exploratory. |
| [Soil grain V4](https://www.kaggle.com/code/krishnayadav456wrsty/soil-grain-v4) | DINO CLS features and within-soil variability | Scaling is fitted before CV; feature-set selection precedes its nested check. |
| [Calibrated CV Ensemble](https://www.kaggle.com/code/avikdas567/soil-grain-size-prediction-calibrated-cv-ensemble) | Another frozen-feature blend | Uses a different trapezoidal scorer and obsolete metadata; ordinary fivefold code is described as nested. |

Public source availability does not settle its license. Reimplement bounded ideas with attribution in our existing fold-safe pipeline, rather than importing whole notebooks.

**A separate visual-language baseline is worth a controlled assay.** A September 7 participant self-reports scores around 34 and 29 from a vision-language model: an unverified anecdote, not evidence of the leading method. Our proposed test must use fresh contexts without held-out labels. An already label-informed assistant conversation is not blind validation. [Participant discussion](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/737406).

**External data is a provenance question before a scaling opportunity.** The public [original Zenodo dataset](https://zenodo.org/records/14725633) has 26 soils, but all 24 current training IDs overlap; its two additional IDs are H031 and H693. It is not 26 new independent examples. Whether its archive preserves useful higher-resolution originals remains unverified. The expanded GRAI3 dataset is described as available on request; it includes German HPC soils and iPhone images, so overlap with our test population must be resolved before using it or a derived checkpoint. Public app source exists, but accessible model weights were not established. No external photos were uploaded or people contacted during this review.

The [competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules) permit external data/models unless specifically prohibited, subject to accessibility, licensing and reproducibility conditions. They are not a guarantee that every proposed resource is appropriate.

**The broader Kaggle lesson is to diagnose the train/test difference before adding models.** Kaggle's first-place IEEE fraud example describes adversarial validation and feature consistency checks; a firsthand BMS write-up similarly investigates different train/test subsets. For our tiny data, a fixed camera-transfer experiment is more interpretable than building a large domain-classification project. [Kaggle solution example](https://www.kaggle.com/solution-write-up-documentation), [BMS solution](https://www.kaggle.com/competitions/bms-molecular-translation/writeups/akirasosa-15th-adversarial-validation-etc).

## Decision for tomorrow

Run a focused diagnostic, then at most two distinct candidate families: **spectral features → Weibull parameters**, and **a fixed, isolated visual-language baseline if available**. Keep square-root mass targets, native-resolution patch distributions, better external data, and limited fine-tuning as subsequent branches, not another simultaneous sweep.

Use **at most two submissions**, leaving at least three of the currently documented five daily slots unused. A slot is permission to test a hypothesis, not a requirement to upload. The [execution plan](superpowers/plans/2026-09-17-domain-and-representation.md) specifies fold boundaries, stopping rules, evidence and handoff. It does not promise a new best or near-zero EMD.
