# October 2 second-round results

The [declaration](october2-round2-plan.md) and producing code were committed as `f7a15c8` before any model request; CI passed. The goal is to beat **30.22376**, set by this morning's first-round all-photos upload, using today's remaining four slots. The user declined leaderboard probing.

## Local evidence

All eight new recipes completed 24 whole-soil holdouts and ten test soils. **273 dispatches; one failed.** That failure was a Codex start-up race in which two clients installed system skills at the same moment (`ERROR … Directory not empty`). Its declared second dispatch succeeded. 55 Codex sessions recovered from WebSocket disconnects under the round-1 transport policy. Gains are against the original VLM's per-soil EMD; positive is better.

| Candidate | Draws | Local LOO EMD ↓ | Soils improved | Median gain | Mean gain excl. top soil |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uploaded all-photos draw (public 30.22376) | 1 | 44.37302 | 16/24 | +2.00 | +3.17 |
| All-photos redraws B / C | 1 each | 47.40603 / 46.90314 | 16 / 14 | +2.27 / +1.08 | +2.19 / +1.32 |
| 150 mm all-photos draws A / B / C | 1 each | 43.13456 / 41.43636 / 41.28393 | 16 / 19 / 19 | +2.58 / +3.40 / +1.38 | +3.74 / +5.45 / +5.80 |
| Claude all-photos draws A / B / C | 1 each | 45.49786 / 46.76183 / 47.76806 | 13 / 13 / 12 | +0.64 / +0.67 / +0.31 | +0.82 / −0.37 / −0.51 |
| **Slot 1: `all_views_draw_mean`** | 3 | 46.10971 | 16 | +1.68 | +2.40 |
| **Slot 2: `all_scales_draw_mean`** | 6 | 43.82860 | 17 | +2.42 | +3.91 |
| **Slot 3: `wide_draw_mean`** | 3 | **41.85984** | 17 | +2.85 | **+5.09** |
| **Slot 4: `claude_all_views_mean`** | 3 | 46.46534 | 13 | +0.42 | +0.20 |

**150 mm crops beat 100 mm in every draw** (41.3–43.1 vs 44.4–47.4). They help 16–19 soils, and the gain survives removing the largest beneficiary. The fresh 100 mm draws suggest the uploaded draw was locally favourable. Averaging improves on the components' mean only slightly, so local error remains mostly systematic. Claude with all photos improves modestly over its single-photo version. Claude cost $5.15 for three recipes. Codex used about 643k input tokens per 100 mm recipe and 766k–789k per 150 mm recipe, all within budget.

## Frozen queue

No candidate was dropped: all are eligible, below the original's 50.62293, and numerically distinct from 32 earlier uploads and from each other. The order is the declared order, fixed before these local results. Selection SHA-256: `31622a32b6e0e7d424e59831e2e2a971104f009eb4ad98faa0180bb8a2bbc220`. Script SHA-256s: `submit.py` `be9beafe…a36b67c`, `poll.py` `eb477fd1…fdf12b`.

| Order | Candidate | Submission SHA-256 |
| ---: | --- | --- |
| 1 | `all_views_draw_mean` | `3d0e6644ce27fcbaf279a076b482dccd75caa08a6736702dd3d60de042d91376` |
| 2 | `all_scales_draw_mean` | `b8f99576f3dd4657715e38ff8362b175d4cd12d6268c5c326fafda3fcb3dd4aa` |
| 3 | `wide_draw_mean` | `3da601ef4988faef399ef23cc5f9f3adbee3545d9e4045b72bf3043ebeb7d9e6` |
| 4 | `claude_all_views_mean` | `b018fb671375904bf2a3a07c9a2d20e2312de2aa5d355e9b22ce4e763e643342` |

## Verification

**657 tests pass.** Independent review before launch caught one Münster test photo whose 150 mm crop shows a ruler and company card. It is excluded from the 150 mm recipes by file hash, as declared.

After the runs, an independent audit with a completeness critic checked:

- every one of the 273 attempts for isolation;
- all 3,482 images, re-rendered byte-for-byte;
- prompts against the declared text, where the 150 mm prompts differ only in the crop sentence;
- that the 100 mm redraws and Claude requests match the uploaded draw's requests exactly;
- every score and every mean (to 1.4e−14);
- distinctness and the freeze/upload logic.

All passed. The same caveats as round 1 apply: three public soils, mutable hosted model aliases, and unverified external-model prize eligibility.
