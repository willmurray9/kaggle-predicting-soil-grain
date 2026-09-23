# September 23 results

The user renewed autonomous work until public EMD beats **37.62858** or the daily allowance is exhausted. The [declaration](september23-plan.md), commit `a248140`, fixed two new visual-context experiments and a saved-candidate queue. Official preflight at **17:17 UTC** found 26 completed submissions, none today, five slots available, and no changes to the seven official pages, 165 data files or discussions.

## Public results — daily allowance exhausted

| Order | Candidate | Submission ref | Local EMD ↓ | Public EMD ↓ |
| ---: | --- | --- | ---: | ---: |
| 1 | Paired-camera examples VLM | `56500720` | 51.29550 | 46.04449 |
| 2 | Complete-context VLM | `56500738` | 40.21564 | 58.45786 |
| 3 | Fixed VLM median | `56500757` | 46.63935 | 43.88009 |
| 4 | Spectral boosting | `56500777` | 44.43476 | 99.24452 |
| 5 | Log-quantile transport ridge | `56500795` | 47.47930 | 75.93323 |

**Stopped after five of five daily slots: all 31 lifetime submissions are complete. Public best remains 37.62858**, original VLM submission `56339475`. Official final state was checked at **17:55:03 UTC**. No further model calls or uploads are scheduled.

The median was today's best at **43.88009**. Neither new visual context improved public EMD. Complete context's much stronger local mean failed to transfer, reinforcing the need to inspect which soils drive a local gain. Spectral boosting and log-quantile transport also lost; those saved alternatives now have public evidence. This round supports retaining the original VLM, while its private performance remains unknown. Predictions and queue order were never changed after public feedback.

## Local experiments and frozen queue

| Order | Candidate | Whole-soil LOO EMD ↓ |
| ---: | --- | ---: |
| 1 | Paired-camera examples VLM | 51.29550 |
| 2 | Complete-context VLM | 40.21564 |
| 3 | Fixed VLM median | 46.63935 |
| 4 | Saved spectral boosting | 44.43476 |
| 5 | Saved log-quantile transport ridge | 47.47930 |

The original VLM scores **50.62293 locally / 37.62858 publicly**. Paired-camera examples retain its eight deterministic training soils but show the first photograph from every available camera for each, grouped with one laboratory CDF. The query still uses one photo per camera. This tests camera appearance variation without changing the model, physical crop size, seed or reasoning setting.

Complete context supplies all **23 other training soils** for each holdout and all **24 training soils** for test queries. Its prompt, single photo per example, 100 mm crops and query views match yesterday's unfinished scientific recipe. All 34 responses were generated afresh. Yesterday's partial outputs were preserved and never reused. Every held-out soil's photos and label remain outside its examples.

Both recipes completed **24 held-out and ten test predictions**. The complete-context local improvement is encouraging; paired-camera examples were slightly worse than the original. The fixed median uses the coordinatewise median of the original, retrieval and multi-scale VLM CDFs after soil-ID alignment. It has no fitted weights or new model calls. The two numerical alternatives retain their exact saved predictions and original producing commits. SVR and ExtraTrees were audited as declared failure replacements but were not needed.

The independent artifact audit reproduced all 68 hosted predictions and the median, reconstructed source crops and prompt groupings, and verified response/receipt accounting. Descriptive gains against the original VLM are below; positive means lower error. Per-soil details are preserved in the audit JSON.

| Candidate | Mean gain | Median gain | Mean gain excluding largest beneficiary |
| --- | ---: | ---: | ---: |
| Paired-camera examples | −0.67257 | −2.46550 | −4.52350 |
| Complete context | +10.40729 | +1.49596 | +4.84529 |
| Fixed VLM median | +3.98359 | +0.14222 | +1.01562 |

Complete context improves 14 of 24 soils and retains a mean advantage after removing its largest beneficiary. However, H549 and H405 together account for **100.22% of its net gain**; excluding both leaves **−0.02498 mean gain**. Its average improvement is therefore concentrated, rather than broad evidence of generalization. These descriptive diagnostics do not change the fixed queue, prove a causal effect separate from hosted-response variability, or establish public/private performance.

Selection verified **3,777 source records** and all **2,306 historical artifact files unchanged**. All five CSVs are numerically distinct from each other and all 26 earlier submissions. Queue and predictions were frozen before any new public score. Selection SHA-256: `6a6130df92ac375d3fcec6321d7dfa52034ca63400ee72f20a99200e71793185`.

| Candidate | Producing commit | Submission SHA-256 |
| --- | --- | --- |
| Paired-camera examples VLM | `b00229bc89b159bfd21adce22e981202970212df` | `83166ef026db8828f69c25865b29e2a0fd82bc1dc9efc275a977a6b6cd5c1f6d` |
| Complete-context VLM | `b00229bc89b159bfd21adce22e981202970212df` | `fa464b9e9c554d7d97f1382b460f08ec8a4d10f4d278c25a2f9b26a9933acdd4` |
| Fixed VLM median | `b00229bc89b159bfd21adce22e981202970212df` | `43f29caaf3d2dfe9f819f0acb5204acc4a2bd2b3ea0ef2c23c80d006b4b125d9` |
| Saved spectral boosting | `02ffc4cbd0a9aa38fad9713214c1fa54212adf1d` | `5ff302deff53a8ee66d1af7c46b9d4733c04b12e653d81eff258d2986eed8d68` |
| Saved log-quantile transport ridge | `6b14133aad80fdd6b4ad797a32e89cbc7ac96377` | `8516dda7e7b6bb8967190b515ddbb752509be78805aa135230a532be78b8e595` |

## Execution and verification

**647 tests passed**, with two existing pandas performance warnings. Independent code review resolved receipt/resume and stderr transport-accounting issues before live inference. [Producing-code CI passed](https://github.com/willmurray9/kaggle-predicting-soil-grain/actions/runs/35896204867). Saved-candidate audits independently reproduced their local scores, validated template alignment and confirmed successful historical producing-code CI.

Paired examples made **34 CLI dispatches**, reporting **665,951 input / 10,877 output tokens**, including **8,871 reasoning tokens**, with zero transport-recovery events or stderr transport warnings. Benign CLI housekeeping warnings were also preserved. Complete context made **34 dispatches**, reporting **804,936 input / 10,320 output tokens**, including **8,314 reasoning tokens**. Eighteen requests had recognized transport recovery: **72 JSON reconnect events and 18 JSON fallback events**, plus **90 stderr retry warnings and 18 stderr fallback warnings**. These overlap and must not be summed as server attempts. All accepted calls ended with exactly one valid completed turn and matching final response; no application-level redispatch, curve repair or response selection occurred.

Combined known completed-turn usage is **1,470,887 input / 21,197 output tokens**, including **17,185 reasoning tokens**. This is below the per-recipe recorded-usage limits. Disconnected server-attempt consumption remains unknown; those limits are not verified provider-side spending caps. No capacity was purchased or reset.

The runner uses Codex CLI **0.154.0**, alias **gpt-6-astra**, low reasoning, fresh anonymous contexts, and disabled tools/apps/memory/repository instructions. Saved responses reproduce the exact CSV, but fresh hosted requests may differ and the alias is not an immutable checkpoint. External-model accessibility and prize eligibility remain unverified. Public performance covers only three soils and does not establish private performance or exclude pretraining overlap. Repeated local validation remains exploratory; camera-transfer performance was not measured for these VLMs or the median.

Code and reports belong in Git. Generated images, prompts, raw events, responses, prediction CSVs, audits, frozen selection and upload receipts remain local under `artifacts/experiments/september23/`, following the existing repository policy.
