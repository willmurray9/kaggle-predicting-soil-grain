# October 6 results

The [declaration](october6-plan.md) and producing code were committed as `f04a22a` before any model request; CI passed. The goal is to beat **30.22376** (`56778295`) or use today's five slots. The user declined leaderboard probing.

## Official results — allowance exhausted, best unchanged

| Order | Candidate | Submission ref | Local EMD ↓ | Public EMD ↓ |
| ---: | --- | --- | ---: | ---: |
| 1 | Example-panel bagged mean | `56885854` | 39.24408 | 36.03861 |
| 2 | Panel + tiled coverage mean | `56885862` | 39.87939 | 36.15340 |
| 3 | Tiled two-draw mean | `56885866` | 42.51267 | 38.77302 |
| 4 | Tiled draw A | `56885873` | 40.93262 | 39.42952 |
| 5 | Tiled draw B | `56885883` | 44.40751 | 38.12980 |

**Stopped: five of five daily slots used; 41 lifetime submissions, all complete. Best remains 30.22376** (`56778295`). The final check was at **16:31:32 UTC**. The queue and predictions never changed after public feedback.

The all-photo GPT family now has nine public scores, from 30.22 to 41.17. Seven of them, all from today and October 2, fall between 33.4 and 39.4; the rest are the 30.22 draw itself and the 41.17 wide-crop mean. Today's local improvements (46.1 → 39.2) did not move public scores below 36. The single 30.22376 draw looks increasingly like a favourable outlier on three public soils rather than the family's typical result.

For final-submission choice: panel bagging gives up 5.81 public EMD against `56778295`. Under `0.30·public + 0.70·private`, it beats that submission if its private EMD is at least **2.49 lower**. Its local whole-soil EMD is **5.13 lower** (39.24 vs 44.37), the first candidate whose local edge exceeds its break-even. Local validation has transferred unevenly, so this is evidence, not proof.

## Local evidence

Six GPT recipes completed 24 whole-soil holdouts and ten test soils: **206 dispatches**. Two first attempts failed, with an identical client error before any response: Codex timed out refreshing its model list (`ERROR codex_models_manager … timeout`). Their declared second dispatches succeeded. A recipe whose second attempt is valid is eligible; this matches the October 2 plan's definition and round 2's precedent. Six test requests recovered from WebSocket disconnects under the transport policy.

Gains are against the original VLM's per-soil EMD and against the seed-0 three-draw all-photo mean (46.11); positive is better.

| Candidate | Local LOO EMD ↓ | vs original: improved / gain excl. top | vs seed-0 mean: improved / gain excl. top |
| --- | ---: | --- | --- |
| Panels 1 / 2 / 3 / 4 (single draws) | 41.529 / 40.152 / 42.366 / 42.609 | 11–14 soils | 9–13 soils |
| Tiled draws A / B | 40.933 / 44.408 | 18 / 16 | 14 / 12 |
| **`panel_bagged_mean`** | **39.244** | 14 / +7.31 | 10 / +4.48 |
| **`coverage_panel_mean`** | 39.879 | 15 / +6.93 | 13 / +4.53 |
| **`tiled_mean`** | 42.513 | 16 / +4.47 | 12 / +2.08 |

These are the VLM family's best local scores. Every alternative panel family (40.2–42.6) beat the seed-0 family (46.1). Seed-0 holdouts mostly reused panels nearly identical to the test panel, so the seed-0 test panel may be a weak one, which panel bagging dilutes. Single panel scores describe each panel's holdout family, not its test panel. Codex used 645k–734k input tokens per recipe, within budget.

## Frozen queue

The order follows the declared rule: means by LOO, then tiled single draws by LOO. No candidate was dropped: all are eligible, below 50.62293, and numerically distinct from 36 earlier uploads and each other. Selection SHA-256: `d7a33b6a42f7574a16c3be87dcc156884d3da7d2d1c333a88f776f88ea64e947`; an auditor's independent simulation reproduced it byte-for-byte. Script SHA-256s: `submit.py` `26f96884…6e0e99`, `poll.py` `1e263f82…f1bf9f`.

| Order | Candidate | Submission SHA-256 |
| ---: | --- | --- |
| 1 | `panel_bagged_mean` | `752a16c7793e0368d819aae2056048eded784d5a199aeed5170eb5a466f0cae4` |
| 2 | `coverage_panel_mean` | `3330aff2f7c3fad0b63b919d17ae033928396da10456e6702d7a23c4f4805b36` |
| 3 | `tiled_mean` | `f3dd4dac10b92cf43cae12afbf959bb25437f20babddd7f9e0a7fd87d2ce1599` |
| 4 | `tiled_all_views_a_vlm` | `11c0850bd70c63ae6d3cfeab8f45c8f64f15f652301b881f296361dd36106ef8` |
| 5 | `tiled_all_views_b_vlm` | `5ffac9926d9df8eccd116a039476789a2cf69172702dae167ec816a44e14e194` |

## Verification

**660 tests pass.** Independent review before launch confirmed the code but corrected two false rationale claims in the draft declaration, before it was committed:

- Seed-0 holdouts mostly share the test panel.
- Round 2's 150 mm recipe widened examples too.

After the runs, an independent audit with a completeness critic checked:

- isolation for every attempt;
- the examples, prompts and tile geometry against independent derivations;
- all 2,958 images, re-rendered byte-for-byte;
- every score and mean;
- distinctness and the freeze/upload logic.

All passed. The public score covers three soils, hosted aliases are mutable, and external-model prize eligibility remains unverified.
