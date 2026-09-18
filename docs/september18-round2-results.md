# September 18 second-round results

The user renewed autonomous work to beat **49.65594 public EMD** or exhaust today's allowance. The [declaration](september18-round2-plan.md) fixes three hypotheses and the upload order before this round's public feedback. Official preflight at 19:42 UTC found 18 completed submissions, two of five used today, three remaining, and unchanged competition pages/data metadata.

**Outcome: new public best 37.62858**, from the unchanged visual-language candidate. This is a **12.02736 EMD / 24.22% reduction** from 49.65594. The first upload met the goal, so this round stopped after one submission. Three of five daily slots are used across both rounds; two remain.

## Local evidence

| Candidate | Whole-soil LOO EMD | Camera disagreement | Soils improved vs native incumbent |
| --- | ---: | ---: | ---: |
| Native spectral incumbent | 41.20828 | 25.77945 | — |
| Native square-root mass ridge | **39.90205** | 23.91515 | **17/24** |
| Native spectral / LBP 50/50 blend | 42.75853 | **21.45848** | 10/24 |
| Existing isolated visual-language model | 50.62293 | Not measured | 11/24 |

Native square-root mass improves mean EMD by **1.30623**, with median per-soil gain **2.18176**. Excluding its largest beneficiary, H405, leaves **0.72682** mean improvement. It uses the same 23 high-resolution features and alpha10 ridge, with the previously declared square-root mass transformation. No parameters are tuned.

The native/LBP blend is worse locally by **1.55025** EMD. Its lower camera disagreement is a secondary diagnostic, not evidence of better unseen-camera accuracy. The two components retain independent fold-fitted scaling and fixed alpha10 ridge; their repaired CDFs are averaged equally. No blend weight is fitted.

Motorola → Samsung / Samsung → Motorola transfer EMD is **45.28945 / 43.59333** for native square-root mass and **46.54693 / 42.47676** for the blend. These use 21 paired soils and exactly 20 other training soils per fold. They do not establish performance on unseen iPhones. Fine-threshold contributions at or below 0.2 mm are **23.30775** and **23.54561** EMD respectively in the pooled holdouts.

The visual-language model reuses the earlier 24 anonymous held-out and ten test responses byte-for-byte. No model call, prompt edit, or example selection occurred in this round. It tests a different representation despite weaker local EMD. Its original isolation, resource accounting and reproducibility limits remain in the [first-round report](september18-results.md).

## Frozen submissions

Upload order is **visual-language → native square-root mass → native/LBP blend**, stopping after any confirmed improvement. This follows the declaration: the distinct visual prior first, then the numerical candidates by local EMD. All three predictions differ from each other and from the 18 earlier uploads. No new public result has been read at selection time.

Selection SHA-256: `2d3c0f77d276e76b1faeaa29b6469a1a2826e923c8e2652786fd61279056608c`.

| Candidate | Producing commit | Submission SHA-256 |
| --- | --- | --- |
| Visual-language | `8c157a7` | `1e2877246a00938044acc139fa7f265f330b6d7bf70824a3e94e0039bed81321` |
| Native square-root mass | `f2da60b` | `740d1babbe944c3eb4f639d277d55af4ede775f4b402ecafe69b8b27c2d263c1` |
| Native/LBP blend | `f2da60b` | `3c453446d85f97b845a4fe3613d176e026c61a48cebb6c4462929bf1fbcba17a` |

The implementation adds one small runner that reuses existing caches, models and whole-soil evaluation. **602 tests passed**, with two existing pandas performance warnings. Independent code review found no actionable defects. All **763 pre-existing artifact files remain unchanged**; new artifacts, selection and receipts belong to `artifacts/experiments/september18_round2/`. Each upload additionally requires successful producing-commit CI and an independent result/provenance check.

## Official result and stop

| Candidate | Submission ref | Uploaded UTC | Public EMD | Change from 49.65594 |
| --- | --- | --- | ---: | ---: |
| **Visual-language** | **`56339475`** | September 18, 19:50:31.483 | **37.62858** | **−12.02736** |

The [official submission](https://www.kaggle.com/competitions/soil-grain-size-from-photos/submissions) is complete. This is the nineteenth lifetime upload and the third today. Native square-root mass and native/LBP remain **unsubmitted** because the goal was met. No further model experiment, prompt revision, blend, or upload follows the result.

The winner is the earlier isolated **gpt-6-astra** visual-language baseline: a pretrained model receives anonymous calibrated soil images and eight training-only labeled examples per query, then returns eleven cumulative percentages. It was not fine-tuned on our dataset and uses no ridge regressor. Exact original prompts/responses and the submission CSV are preserved. This round reused those predictions without additional model requests.

Its public score improves substantially despite worse local EMD than the native spectral model (**50.62293 versus 41.20828**). The public result covers only three soils; it does not establish private-set superiority, explain the causal source of improvement, or eliminate pretraining-overlap uncertainty. The result supports further investigation of a different visual prior, without changing the stopped round's recipes.

Both numerical candidates' [producing-code CI jobs passed](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35387864093), as did the submitted VLM's [original CI](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35376569929). Independent audits recomputed every new local score, checked all 168 camera-transfer fold records, confirmed the fixed blend against the original components within 2.46e−13 percentage points, and verified original VLM contexts and artifacts remain unchanged. The selection, accepted-reference receipt, audits, final status and artifact inventory are saved in the round's output directory.
