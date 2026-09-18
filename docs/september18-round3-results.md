# September 18 third-round results

The user renewed autonomous work to beat **37.62858 public EMD** or exhaust the remaining daily allowance. The [declaration](september18-round3-plan.md), committed as `d61ad58` before evaluation, fixes two hypotheses. Official preflight at **20:25:51 UTC** found nineteen completed lifetime submissions, three of five used today, two remaining, and unchanged official pages/data metadata.

**Outcome: both uploads completed without beating 37.62858. All five daily slots are used, so the round stopped.** There are twenty-one complete lifetime submissions and none pending.

## Local evidence

All validation holds out an entire physical soil, including its photos. No prompt, blend-weight, neighbor-count or feature search was performed in this round.

| Candidate | Whole-soil LOO EMD ↓ | Role |
| --- | ---: | --- |
| Original isolated visual-language model | 50.62293 | Public incumbent: 37.62858 |
| Saved native square-root mass ridge | 39.90205 | Blend component; not submitted alone |
| **50/50 original VLM / native mass blend** | **43.05461** | First queued upload |
| **Similarity-selected visual-language examples** | **43.11100** | Second queued upload if needed |

The blend averages the two saved CDFs at exactly 50/50 after aligning soil IDs. Relative to the original VLM, it improves **15/24 soils**, with mean gain **7.56833**, median gain **2.50550**, and mean gain **5.57088** after excluding its largest beneficiary. It is worse on average than the native mass component by **3.15256**. The components' residual correlation is about 0.73, so their complementarity is limited. No new model requests are needed for this candidate.

The second candidate replaces the original random eight examples with the eight closest labeled soils under the existing 23 native spectral features. Features are averaged within soil; standard deviations are fitted using the other 23 soils for validation, or all 24 labeled soils for test queries. Query labels never enter selection or prompts. Ties use sorted soil IDs, and examples appear in distance order. The model, low reasoning, task instructions, 100 mm crop preparation and number of example images match the original VLM.

All **34 isolated requests** completed with valid curves: 24 whole-soil holdouts and ten test soils. They consumed **590,928 input tokens and 9,354 output tokens**, including **7,348 reasoning tokens**, within the declared one-million-input/50,000-output aggregate limits. There were no retries, repaired responses or omitted holdouts.

Retrieval improves **14/24 soils**, with mean gain **7.51193** and median gain **1.26746** against the original VLM. H549 accounts for about 74% of the total gain; excluding it leaves a smaller positive mean gain of **2.03483**. This supports the example-selection hypothesis but does not isolate its causal effect from variability in fresh hosted-model responses. Camera-transfer performance was not measured for either new candidate. Their local difference is only **0.05640 EMD**; this determines the declared ordering without establishing a reliable generalization ranking.

## Frozen submissions

The declaration orders the two eligible experimental outputs by local EMD: **blend → similarity-selected VLM**, stopping on the first confirmed public improvement. The native mass fallback was not needed. Both CSVs differ numerically from each other and all nineteen previous submissions. The selection was frozen before new public feedback.

Selection SHA-256: `45a9519fc6cd100ffd85701c233fcceacf308e2786635e4cb8e5e283f8befbc9`.

| Candidate | Producing commit | Submission SHA-256 |
| --- | --- | --- |
| VLM / native mass blend | `91ddcac9db763432afbbbf2c9c7d93077c8ae785` | `974077962adf56a501cf46fc756dc044e94988369f27df1aad13e16ecdd1397c` |
| Similarity-selected VLM | `0d65e493a3ce14b560266e8a47c51d1582015565` | `89a33d7b476cc47f9fe25d0c7d997a97f9faac72a5c12c62a85b9ca1a769f946` |

**621 tests passed**, with two existing pandas performance warnings. Producing-code CI passed for both the [blend](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35391830296) and [retrieval](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35392318779). Selection verified 993 source records and all **788 pre-existing artifact files unchanged**. Prompts, prepared images, raw events, responses, immutable request records, audits, selection and upload receipts are saved locally under `artifacts/experiments/september18_round3/`; generated artifacts remain outside Git.

Independent result audits reconstructed the blend from its saved components, recomputed both local scores, checked every retrieval example/scaler/prompt, and reconstructed all **327 prepared images byte-for-byte**. Request receipts precede model dispatch; upload receipts precede Kaggle dispatch. No uncertain request was retried.

## Reproduction limits

The retrieval run uses authenticated Codex CLI **0.154.0**, model alias **gpt-6-astra**, low reasoning, and fresh anonymous contexts with tools, apps, memory and repository instructions disabled. Saved responses reconstruct the exact CSV. The hosted alias is not an immutable checkpoint, so rerunning fresh requests is not guaranteed to reproduce identical predictions. External-model accessibility and organizer acceptance of this arrangement remain unverified; CSV acceptance alone does not establish prize eligibility. No service capacity was purchased or reset.

The public score covers three soils. It does not establish private performance or resolve pretraining-overlap uncertainty.

## Official results and stop

| Order | Candidate | Submission ref | Uploaded UTC | Public EMD ↓ | Change from 37.62858 |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | VLM / native mass blend | `56340871` | September 18, 20:47:06.047 | 44.06175 | +6.43317 |
| 2 | Similarity-selected VLM | `56340908` | September 18, 20:48:24.373 | 47.32415 | +9.69557 |

The [official submissions](https://www.kaggle.com/competitions/soil-grain-size-from-photos/submissions) are complete. Fresh status at **20:49:26 UTC** confirms **five of five daily slots used**, twenty-one complete lifetime uploads, and the original visual-language submission `56339475` still best at **37.62858**. No new model calls, fitted variants or uploads follow these results. The saved standalone native mass fallback remains unsubmitted.

Both candidates improved local EMD but worsened the public score. The blend reduced local VLM errors while diluting its public advantage; retrieval's local gain was concentrated in H549 and did not carry over to the three public soils. These are observations, not proof of which visual cues helped the original model. Keep the original winner and the fixed failed hypotheses intact. Any next round needs a fresh declaration; no further work is scheduled.
