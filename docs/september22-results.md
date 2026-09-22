# September 22 results

The user renewed autonomous work until a public improvement over **37.62858** or exhaustion of the daily allowance. The [declaration](september22-plan.md), commit `18fb903`, fixed two information-focused VLM experiments and saved numerical replacements. Official preflight at **15:07:54 UTC** confirmed twenty-one completed submissions, none today, five slots available, and unchanged official pages and data.

## Public results — daily allowance exhausted

| Order | Candidate | Submission ref | Local EMD ↓ | Public EMD ↓ |
| ---: | --- | --- | ---: | ---: |
| 1 | Multi-scale VLM | `56468669` | 48.66290 | 40.88127 |
| 2 | Native square-root mass ridge | `56468686` | 39.90205 | 54.55433 |
| 3 | LBP ridge | `56468706` | 45.57380 | 64.42273 |
| 4 | EMD median regression | `56468726` | 48.58551 | 72.33884 |
| 5 | Native spectral / LBP blend | `56468742` | 42.75853 | 53.72754 |

**Stopped after five of five daily slots: all twenty-six lifetime submissions are complete. Public best remains 37.62858**, original VLM submission `56339475`. Official state was verified at **15:49:45 UTC**. No further model requests or uploads are scheduled.

The multi-scale VLM was today's strongest candidate at **40.88127**, but its local improvement did not transfer to a public improvement. The four saved numerical alternatives also failed to beat the incumbent. We did not change predictions or queue order after observing these scores. This round provides no evidence to replace the original VLM; the complete-context hypothesis remains untested because its run failed.

## Local results and frozen queue

| Order | Candidate | Whole-soil LOO EMD ↓ | Source |
| ---: | --- | ---: | --- |
| 1 | Multi-scale VLM | 48.66290 | New paired 100 mm / 25 mm images |
| 2 | Native square-root mass ridge | 39.90205 | Saved September 18 predictions |
| 3 | LBP ridge | 45.57380 | Saved September 18 predictions |
| 4 | EMD median regression | 48.58551 | Saved September 18 predictions |
| 5 | Native spectral / LBP blend | 42.75853 | Declared replacement for failed VLM |

The original VLM scores **50.62293 locally / 37.62858 publicly**. The multi-scale recipe retains its eight deterministic training examples and low-reasoning model. It adds a calibrated 25 mm center crop beside each original 100 mm crop, including every query view. Both come directly from the source photograph; neither is enlarged. A 25 mm test crop retains 349–488 native pixels per side, compared with about 114–115 for most training photos. This provides more test detail while retaining the mismatch in source resolution; it does not resolve fine grains in low-resolution training images.

All **24 whole-soil holdouts and ten test predictions** completed for multi-scale. Its mean local gain over the original VLM is **1.96004 EMD**. Camera-transfer performance was not measured. The comparison remains exploratory and cannot separate the added image information from fresh hosted-response variability.


The gain is concentrated: **10 of 24 soils improved**, the median gain is **−1.22559**, and excluding the largest beneficiary, **H549 (+71.84645)**, the mean gain is **−1.07850**. Positive gain means lower error. These diagnostics weaken the case for a broad improvement; they do not change the predeclared submission queue.

| Soil | Original VLM EMD | Multi-scale EMD | Gain ↓ error |
| --- | ---: | ---: | ---: |
| F827 | 66.51271 | 70.13695 | -3.62424 |
| G190 | 42.90428 | 38.87701 | +4.02728 |
| H030 | 33.70717 | 40.20615 | -6.49899 |
| H037 | 99.10149 | 96.94828 | +2.15321 |
| H038 | 123.59436 | 124.99707 | -1.40270 |
| H126 | 61.86402 | 44.03914 | +17.82488 |
| H181 | 20.11622 | 17.82028 | +2.29595 |
| H183 | 15.41091 | 13.68330 | +1.72762 |
| H366 | 5.33638 | 7.43807 | -2.10169 |
| H367 | 9.47990 | 15.38058 | -5.90068 |
| H368 | 21.42563 | 22.87479 | -1.44916 |
| H371 | 9.48403 | 13.77706 | -4.29302 |
| H372 | 16.53847 | 17.89235 | -1.35389 |
| H374 | 26.94575 | 32.55082 | -5.60507 |
| H405 | 165.07207 | 151.87308 | +13.19899 |
| H493 | 77.51525 | 71.17184 | +6.34341 |
| H516 | 49.76227 | 50.85957 | -1.09730 |
| H549 | 148.75306 | 76.90661 | +71.84645 |
| H615 | 6.12123 | 3.11515 | +3.00608 |
| H616 | 84.16078 | 91.36315 | -7.20237 |
| H617 | 12.80260 | 15.80260 | -3.00000 |
| H637 | 39.74848 | 34.55186 | +5.19662 |
| H666 | 20.79155 | 21.64341 | -0.85186 |
| H668 | 57.80175 | 94.00040 | -36.19865 |

The complete-context recipe supplies all other 23 labeled soils in each holdout and all 24 for test queries. It failed its execution check after two CLI dispatches, so it has **no eligible local score or submission**. Its partial responses were preserved and not reused. The five-candidate queue above follows the original declaration, including its fixed failure replacement. Spectral boosting was audited as the second replacement but was not needed. Numerical candidates were neither refitted nor altered.

All five frozen CSVs are distinct from each other and all twenty-one earlier uploads. Selection verified **1,314 source records** and all **1,348 historical artifacts unchanged**. Selection SHA-256: `bf3ef61c046fb1d98f4a8060957560813faa862cb091895ca5dc6c705b29c29a`.

| Candidate | Producing commit | Submission SHA-256 |
| --- | --- | --- |
| Multi-scale VLM | `ec82bde3e22458fcfa2516b73396b4f278d80f95` | `62c44f06a7f3959dc28dcd684ba9a5b4def12ecc68267358e69491aa3ebc6104` |
| Native mass ridge | `f2da60ba8337759cfd8df1e07ecadad7fe612710` | `740d1babbe944c3eb4f639d277d55af4ede775f4b402ecafe69b8b27c2d263c1` |
| LBP ridge | `8d3adaacb450e1ee18aa08bf9791df73770050f6` | `2541d1b28c78549a82e8c8881683fb7aeaae8a5d703748d3df35028104dfae86` |
| Median regression | `8d3adaacb450e1ee18aa08bf9791df73770050f6` | `de5fdf47c1e0af8179e60bc2117e6fe0c8ad72e821abce94a832538575dae1fc` |
| Native/LBP blend | `f2da60ba8337759cfd8df1e07ecadad7fe612710` | `3c453446d85f97b845a4fe3613d176e026c61a48cebb6c4462929bf1fbcba17a` |

## Execution deviation and resource accounting

The CLI performed internal transport retries, contrary to the declaration's literal blanket no-retry wording. This was [recorded during inference](september22-execution-note.md), before complete local evaluation or new public feedback. The precommitted event validator was retained: the complete-context G190 request contains reconnect error events and is rejected; the multi-scale H030 request contains one stderr retry warning, no error event and one valid completed turn, and is retained. There was no application-level redispatch, response repair, selection among completed outputs or prompt change.

Multi-scale made **34 CLI dispatches**, reporting **619,449 input / 10,136 output tokens**, including **8,132 reasoning tokens**. Complete-context made two dispatches; their completed events report **47,091 input / 645 output tokens**, including **527 reasoning tokens**. The latter includes G190's known usage omitted from its failed ledger entry because validation raised first; the ledger was preserved unchanged.

The combined known completed-event total is **666,540 input / 10,781 output tokens**, including **8,659 reasoning tokens**. These are not verified totals for all underlying server attempts: disconnected-attempt consumption is unknown. Recorded usage is below the aggregate limits, but it does not prove a strict provider-side spending bound. No capacity was purchased or reset.

## Verification and reproduction

**632 tests passed**, with two existing pandas performance warnings. Independent review checked crop fidelity, example/query exclusion, prompt layout, response validation, receipt ordering, budgets, and safe interruption/resume behavior. The [new producing-code CI passed](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35747184152), as did the saved candidates' producing commits. The independent artifact audit reconstructed all 34 multi-scale requests, checking source-photo crops, exclusion of held-out soils, prompts, records, raw events, responses, exported predictions and token totals. Independent numerical audits reproduced all saved local scores, verified valid ten-soil submission curves, and checked source hashes and historical preservation.

Code and reports are committed to Git. Generated prompts, images, raw events, responses, immutable request records, candidate CSVs, selection, upload receipts and audit evidence are local under `artifacts/experiments/september22/`, following the existing repository policy.

The runner uses Codex CLI **0.154.0**, alias **gpt-6-astra**, low reasoning, and fresh anonymous contexts with tools/apps/memory/repository instructions disabled. Saved responses reconstruct the exact CSV; a new hosted run is not guaranteed identical. The alias is not an immutable checkpoint. External-model accessibility and organizer acceptance remain unverified, and successful scoring does not establish prize eligibility. The three public soils do not establish private-set performance or resolve pretraining-overlap uncertainty.
