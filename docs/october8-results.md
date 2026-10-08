# October 8 results

The [declaration](october8-plan.md) and producing code were committed as `f57c47b` before any model request; CI passed. This round tests whether the October 7 matched-resolution result (32.70180) replicates. The goal remains to beat **30.22376** (`56778295`) or use today's five slots. The user declined leaderboard probing.

## Local evidence

Six fresh matched-resolution draws (c–h) completed 24 whole-soil holdouts and ten test soils.

- **210 dispatches.** Six first attempts timed out after 300 s, two each in c, d and e, all in the first batch at the same queries. Each was charged the declared ceiling and succeeded on its second dispatch.
- Every request rebuilt October 7's matched requests byte-for-byte.
- Every response carried only the declared managed-requirements notices.

As expected for a test-only change, local whole-soil EMD sits within the base recipe's noise.

| Candidate | Draws | Local LOO EMD ↓ |
| --- | ---: | ---: |
| Single draws c / d / e / f / g / h | 1 each | 47.421 / 46.561 / 46.566 / 48.285 / 50.730 / 46.040 |
| **`matched_new_six_mean`** (replication test) | 6 | 47.314 |
| **`matched_eight_draw_mean`** (final candidate; includes a, b) | 8 | 47.169 |
| **`matched_cd_mean` / `ef` / `gh`** | 2 each | 46.835 / 47.425 / 48.176 |

On the test soils, the six-new-draw mean averages 4.75 EMD from October 7's `matched_mean` and 9.74 from the 30.22 winner. Codex charged 4.01M input tokens, including 240k of timeout ceilings, within budget.

## Frozen queue

The order is the declared fixed order. All five candidates are eligible and below 50.62293, and are distinct from 46 earlier uploads and each other. The single draw g (50.73) exceeds the threshold as a standalone fallback; the rule applies only to queued candidates, so g remains a declared component of the means. Selection SHA-256: `e99dfa512f5430a2c5d7fab0f9d353e83c4b26d89bd21637a77d63e5971b7048`. Script SHA-256s: `submit.py` `58e5d2d2…502de3`, `poll.py` `c28a8d74…0db40`.

| Order | Candidate | Submission SHA-256 |
| ---: | --- | --- |
| 1 | `matched_new_six_mean` | `ad716d60844eb1c6cfde4cee49c49ec65f4d3811f71909d141716204d3ee1895` |
| 2 | `matched_eight_draw_mean` | `648d46293cb8ae52bd95dd012f8214725cc129733561d83a15b88daa475370ea` |
| 3 | `matched_cd_mean` | `3d9c27719b276a61ab2d1aa4b1c0812cf81d54578b01045f7b9cb0bbb61e97ed` |
| 4 | `matched_ef_mean` | `8a63adb53b7a8bc601cf962783a6c8feb7725cd3a55f651f88a12ddda566d8ba` |
| 5 | `matched_gh_mean` | `386a2fa4ab80429e96a00c28af381591e0155bb1ca1bced8145c8bb2898e7c75` |

## Verification

**664 tests pass.** Independent review before launch confirmed the code and corrected two points, both before the plan was committed:

- the freeze script now requires every candidate to have finished as exactly one of eligible or failed;
- the interpretation rule now says one three-soil comparison cannot establish a benefit.

The pre-upload audit with a completeness critic checked:

- all 210 attempts for isolation and declared-notice-only streams;
- all 2,681 images, re-rendered byte-for-byte against October 7;
- that the six timeouts were legitimate retries;
- every score, mean and distinctness check, and the upload logic.

All passed.
