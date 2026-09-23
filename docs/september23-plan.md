# September 23 experiment declaration

The user renewed autonomous work until a confirmed public EMD beats **37.62858** or today's allowance is exhausted. Official preflight at **17:17 UTC** confirms 26 completed submissions, none today, five available, unchanged rules/pages and all 165 data files unchanged. This is a bounded continuation of the existing pipeline under the user's standing authorization. Preserve all 2,306 historical artifact files and finish tested work on clean, pushed `main` with the personal Git identity.

## Two new visual-context experiments

1. **`paired_examples_vlm`: teach camera variation explicitly.** Keep the original deterministic eight training soils, original 100 mm physical crops, model and low reasoning. For each example soil, provide the first photograph from each available camera, grouped together with one shared laboratory CDF. Query views remain the first photo per camera. Sort cameras and paths as before. This changes the information supplied for each example, not the examples' labels or retrieval rule.
2. **`complete_context_vlm`: finish the untested full-context hypothesis.** Supply all other 23 training soils per holdout and all 24 for test queries, alphabetical ID order outside the anonymous prompt, with the first original 100 mm crop per example. Reuse the September 22 full-context scientific prompt and image recipe. This is a fresh run under today's transport policy; preserve yesterday's failed run and never reuse its responses.

For both, exclude every photo and label of the held-out soil from examples. Use fresh anonymous Codex CLI 0.154.0 contexts, alias `gpt-6-astra`, low reasoning, and the original disabled-tool/app/memory/repository settings. No model, prompt, seed, example-count, weight or reasoning search. Each recipe requires all 24 whole-soil held-out and ten test predictions. Save prompts, original-source hashes, rendered images, raw events, responses and a receipt before every CLI dispatch. Producing code and this declaration must be committed before requests.

Allow at most **34 application dispatches, 2,000,000 reported input tokens and 50,000 reported output tokens (including reasoning) per recipe**; check cumulative usage before each next dispatch and after completion. Each dispatch times out after 300 seconds. Do not redispatch a failed, uncertain or malformed request, repair its curve or drop a holdout. A failed recipe is ineligible. A completed ordered prefix may resume only after its identity, sources and records verify.

**Transport policy, declared before inference:** the CLI may internally reconnect or fall back from WebSockets to HTTPS. Retain the raw warnings and accept only the recognized reconnect/fallback messages if the process succeeds, exactly one turn completes, exactly one final message matches the response file, and the usual no-tool and numeric-CDF checks pass. Reject other errors, failed turns, multiple completed outputs or tool activity. Count and disclose these transport events. This policy does not select among predictions and does not rewrite raw logs. Reported completed-turn usage cannot establish consumption of disconnected server attempts; do not claim a strict provider-side spending bound. No capacity purchase or reset.

## Fixed saved candidates and queue

Compute one deterministic **`vlm_median`**: align the saved original, retrieval and multi-scale VLM predictions by soil ID, then take their coordinatewise median, for both 24 OOF and ten test rows. No fitting or chosen weights. Median preserves monotone CDFs and tests whether disagreements among visual contexts contain isolated extremes. Pin each source and preserve its original whole-soil fold.

The primary queue is **paired examples → complete context → VLM median → saved spectral boosting → saved log-quantile transport ridge**. If a new VLM fails, or a candidate is numerically identical to an earlier upload or queued CSV, omit it and fill the vacant slots at the end with saved nested linear SVR, then saved shallow ExtraTrees. Those saved numerical candidates are reused byte-for-byte; no refits. Today's authorization replaces their historical no-upload selection gates, not their historical records. Integrity failures require resolution rather than a silent replacement.

Finish the candidate pool and freeze at most five CSV hashes and their order before reading any new public result. Local scores and per-soil gains are descriptive and cannot change the declared queue. Report median gain and mean excluding the largest beneficiary against the original VLM for new VLMs and the median. No new camera-transfer claim.

Before every upload verify the current UTC day, quota, public best, exact known history, completed prior requests, producing-code CI, source fingerprints, CSV template order, valid CDFs and numerical distinctness. Save the request before uploading, resolve accepted or uncertain references, and stop immediately upon a confirmed improvement or exhausted allowance. Do not modify remaining predictions after public feedback.

## Preservation and limits

Add only a small new runner and focused tests; historical producing modules stay unchanged. New generated artifacts belong under `artifacts/experiments/september23/` and retain the existing local-only policy. Commit code and reports, merge to `main`, push and delete the fully merged task branch. Include official final status, independent audit and artifact inventory.

Saved hosted responses reconstruct the CSV; a fresh run need not match. The model alias is not an immutable checkpoint, and external-model accessibility/prize eligibility remain unverified. Prior scoring does not resolve that issue. The public score covers only three soils, so a public win would not establish private-set performance. No future run or automatic upload is scheduled.
