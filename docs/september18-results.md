# September 18 experiments and submissions

The user authorized continued work until public EMD improves on **55.78511** or the five daily submissions are exhausted. The [execution declaration](september18-plan.md) supersedes the earlier two-submission cap. Fresh preflight found no data or rules changes, no submissions today, and the same incumbent.

## Camera diagnostic

Each row averages 21 held-out soils; each fit uses exactly the other 20 paired soils from one camera. All photos of the held-out soil are excluded. Lower EMD is better.

| Model | Motorola → Motorola | Motorola → Samsung | Samsung → Motorola | Samsung → Samsung |
| --- | ---: | ---: | ---: | ---: |
| RGB13 ridge | 44.8927 | 48.5747 | 44.0585 | 47.1405 |
| Spectral23 ridge | 43.0596 | 43.9674 | 41.0573 | 46.8408 |
| DINO PLS1 | 39.7641 | 48.3078 | 41.9920 | 46.8467 |

For a transfer penalty, compare against the control photographed by the **same query camera** as well as the declared same-source-camera control. DINO's Motorola → Samsung error is 8.54 above Motorola → Motorola, but only 1.46 above Samsung → Samsung. The first difference also includes query-camera difficulty. DINO's reverse transfer is 2.23 worse than its Motorola control. Spectral transfer beats the matched query-camera controls in both directions, by 2.87 and 2.00 EMD.

This provides no consistent known-camera penalty for our incumbent, and weakens that explanation for the entire public gap. It does not establish performance on unseen iPhones, geological sources, or the three public soils. Saved pooled 24-soil scores use different folds and remain separate.

## Weibull capacity check

The fixed two-parameter family reconstructs the known labels at **8.23074 mean / 18.91703 maximum EMD**, failing the predeclared ≤5 mean / ≤15 maximum gate. G190 has 18.91703 reconstruction error and H126 has 16.54545. No Weibull image regressor or submission is produced. This is representation approximation error, not predictive validation.

## Five fixed candidates

All numerical models use whole-soil LOO with training-only scaling and the original metric. Each changes one target/objective or image representation; no parameters or blend weights are searched.

| Frozen upload order | Candidate | Local EMD | Camera disagreement | Soils improved vs spectral |
| ---: | --- | ---: | ---: | ---: |
| — | Spectral23 incumbent | 40.53927 | 25.33273 | — |
| 1 | Square-root mass ridge | **39.86747** | **22.86752** | **15/24** |
| 2 | Native-resolution spectral ridge | 41.20828 | 25.77945 | 10/24 |
| 3 | Local binary pattern ridge | 45.57380 | 26.28702 | 9/24 |
| 4 | Median CDF regression | 48.58551 | 36.01612 | 7/24 |
| 5 | Fixed visual-language baseline | 50.62293 | Not measured | Recorded separately below |

Square-root mass ridge improves mean EMD by **0.67180**, with median per-soil gain **1.82168**. Excluding its largest beneficiary leaves mean gain **0.07219**: positive but small. Its Motorola → Samsung / Samsung → Motorola errors are **41.179 / 40.117**, better than spectral's **43.967 / 41.057**. It is the strongest first upload. Camera disagreement remains a secondary diagnostic, not direct transfer accuracy.

The other candidates are weaker locally. Their uploads, if needed, test distinct hypotheses: retaining image detail, brightness-order texture distributions, absolute-error regression, and broader visual priors. Median regression has particularly poor Motorola → Samsung transfer (72.434 EMD); its public check is exploratory, not evidence of a robust improvement. The declared order follows local EMD, and public feedback will not change recipes or later predictions.

The visual-language model is `gpt-6-astra` with low reasoning through an already authenticated Codex CLI, using 8 deterministic training-only examples per query. All 24 holdouts and 10 test requests return valid curves in fresh ephemeral, tool-disabled contexts. Images and prompts are anonymous; query labels are absent. The 2 synthetic checks plus 34 soil requests consume **610,790 input / 9,261 output tokens**, including a **7,203-token reasoning subset**. There is no additional camera-transfer evaluation. The model alias is not an immutable provider revision, and pretraining overlap with public data is unknown; exact future replay is not guaranteed. All raw responses and prompt/image hashes are saved locally.

## Submission selection and provenance

The queue was frozen before any September 18 public result. Selection SHA-256:
`66db2889b7ef6f969a954fabe650af7234a001e667bbb49a4ceca28ce38a7474`.

All five predictions differ from one another and from the sixteen previous submissions. Before each upload, verify the current quota/best, input/output hashes, template order, producing commit and successful CI. Save a request receipt before calling Kaggle and resolve the exact reference before another upload. Stop on a new best or quota exhaustion.

- Initial declaration: `7795916`; initial producing code: `8c157a7`.
- Follow-up declaration: `9bcd718`; follow-up producing code: `8d3adaa`.
- Verification: **596 tests pass**, with two existing pandas performance warnings; independent reviews cover fold boundaries, numerical behavior, source provenance, cache identity, and isolated model requests.
- Outputs: `artifacts/experiments/camera_transfer/`, `weibull/`, `visual_language/`, `distribution_search/`, and `september18/`.
- All 243 artifacts present before this session remain unchanged.

## Official results

Pending the frozen sequential submissions. No public score is inferred from local validation.
