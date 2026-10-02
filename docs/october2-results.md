# October 2 results

The user asked for a review of the work so far, new ideas, and autonomous work until public EMD beats **37.62858** or today's five uploads are used. The [declaration](october2-plan.md) and runner were committed as `dc05595` before any model request. Preflight at **15:57 UTC** found 31 completed submissions, none today, five slots, and a November 30 deadline.

## Review and ideas

The visual-language (VLM) family has every one of our seven best public scores: 37.6, 40.9, 43.9, 44.1, 46.0, 47.3 and 58.5. The numerical models' best is 49.7. Every earlier VLM variant changed the model's context and lost. This round therefore kept the winning recipe and changed one thing per candidate. The review found three concrete problems in how test soils were shown to the model:

- **One photo per test soil.** The original takes the first photo per camera. Training holdouts usually have two cameras. Each test soil has one iPhone camera, so the model saw **one of its 3–5 photos**: a single 100 mm square, even for the cobble and gravel soils.
- **Sharper test crops.** The labelled examples are ~460 px crops (~4.6 px/mm). iPhone queries were 768 px, downsampled from 1394–1953 px.
- **Noisy draws.** A fresh rerun of the identical original recipe differs from the saved draw by **8.9 EMD per holdout and 7.1 per test soil** on average. EMD is convex in the prediction, so averaging draws cannot raise expected error.

Higher reasoning effort and a second model family (Claude Opus 5.5) were the remaining untested levers.

**Considered but not pursued:** leaderboard probing. Public scores of 0.00001 and 0.09 now top the leaderboard, almost certainly from probing the three public soils. Our 31 scored submissions would allow similar inference, but repository policy forbids tuning to public cases. This option is the user's decision. External data containing test soils, and hand-written curves, were also excluded.

## Official result — new best, stopped after one upload

| Order | Candidate | Submission ref | Local EMD ↓ | Public EMD ↓ |
| ---: | --- | --- | ---: | ---: |
| 1 | **All query photos** | **`56778295`** | 44.37302 | **30.22376** |

**New public best 30.22376**, a **7.40482 EMD / 19.68% reduction** from 37.62858. The upload completed at **16:46:12 UTC**. The first upload met the declared goal, so the round stopped. **One of five daily slots was used** and 32 lifetime submissions are complete. The other four frozen candidates remain unsubmitted, and no prediction or queue changed after the result.

## Local evidence

Each recipe ran 24 whole-soil holdouts and 10 test soils. All **204 model requests were valid on their first attempt**, with no retries, transport recoveries or tool use. Gains are against the original VLM's per-soil EMD; positive is better.

| Candidate | Local LOO EMD ↓ | Soils improved | Median gain | Mean gain excl. top soil |
| --- | ---: | ---: | ---: | ---: |
| Original VLM (public 37.62858) | 50.62293 | — | — | — |
| **All query photos** | **44.37302** | **16/24** | **+2.000** | **+3.170** |
| Three-draw mean | 50.23001 | 11/24 | −0.093 | −1.260 |
| Matched resolution | 50.90432 | 10/24 | −0.601 | −2.433 |
| High reasoning | 51.23227 | 12/24 | +0.124 | −3.632 |
| Claude Opus 5.5 | 49.99767 | 14/24 | +2.309 | −2.456 |
| Rerun A / Rerun B | 49.62497 / 50.90845 | 9 / 10 | −0.631 / −0.351 | −1.643 / −2.624 |

**Showing every query photo is the only change with a broad local gain**: two-thirds of soils improve, and the gain survives removing the largest beneficiary. The other changes are within draw noise. The draw mean improves only slightly on single draws, so local error is mostly systematic rather than sampling noise. Higher reasoning used about 1.8× the output tokens without a gain. Claude cost **$1.37** for 34 requests.

Resource use: about 600k Codex input tokens per standard recipe and 643k for all-views; 9.3k–18.1k output tokens; Claude 177k input tokens. All are within the declared budgets.

## Frozen queue

The queue below is the declared order. No candidate was dropped: all were eligible, at most 60.0 locally, and numerically distinct from the 31 earlier uploads and from each other. Selection SHA-256: `7699a51a59fe19120feb3e92b22f5095518903fdbdf398db418adb91e7db2e22`. Every candidate was produced by `dc05595`, whose CI passed.

| Order | Candidate | Submission SHA-256 |
| ---: | --- | --- |
| 1 | `all_views_vlm` | `199dd51b0e6eb031571be561d5e475242d6c08a334773f2f7fef0cdefcbc4119` |
| 2 | `vlm_draw_mean` | `78187a7c8ffad05e32097ae12745d5b61c7ee66460b0cb2057c1a87c376b0081` |
| 3 | `matched_resolution_vlm` | `688144af350bb679e82bc45ec3bec1d795fe8f6a0e3aad0fec589793e1246cfa` |
| 4 | `high_reasoning_vlm` | `f3a74be9230dc7c2e3ac649bacedb6b2d187f6b9dc9ca662a39ae134c56d43f9` |
| 5 | `claude_vlm` | `06f0dfb142dd2f5ae18588db6f48b07010e127d4dd59dbc3d2f40caf87cd0ec9` |

## Verification

**655 tests pass**, with two existing pandas performance warnings. Before any request, an independent three-lens code review confirmed six defects, all fixed:

- unbudgeted failed attempts;
- Ctrl-C and launch errors consuming attempts;
- two resume edge cases;
- an underspecified draw mean.

After the runs, an independent audit with a completeness critic:

- re-derived every prompt and photo selection, and re-rendered all 2,069 crops byte-for-byte;
- confirmed no held-out photo, label or soil identifier reached any request;
- confirmed the reruns, high-reasoning and Claude requests reproduce the original 34 prompts and images exactly;
- recomputed every LOO score and CSV row;
- confirmed Claude saw the images (its input tokens track image area, r = 0.998).

One isolation caveat applies to the unsubmitted Claude candidate only. Its built-in AGENTS.md plugin may have loaded an 804-byte user-level `AGENTS.md`. That file contains no soil content, the token overhead was constant across queries, and the only available tool was the structured-output schema.

The audit found that kaggle 2.2.2 reports a failed upload as reference 0 without raising. The upload script now verifies nothing was created and sets the receipt aside, rather than recording a false acceptance.

The public score covers three soils and cannot establish private performance. The hosted model aliases are not immutable checkpoints, and external-model prize eligibility remains unverified. Generated prompts, images, responses, CSVs and receipts remain local under `artifacts/experiments/october2/`.
