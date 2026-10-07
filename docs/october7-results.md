# October 7 results

The [declaration](october7-plan.md) was committed as `6e5a910` before any model request. The goal is to beat **30.22376** (`56778295`) or use today's five slots, testing whether test queries rendered like the labelled examples help. The user declined leaderboard probing.

## Official results — allowance exhausted, best unchanged

| Order | Candidate | Submission ref | Local EMD ↓ | Public EMD ↓ |
| ---: | --- | --- | ---: | ---: |
| 1 | Aligned mean (460 px + exposure) | `56917684` | 47.83571 | 39.98396 |
| 2 | **Matched-resolution mean (460 px)** | `56917695` | 46.77172 | **32.70180** |
| 3 | Exposure mean | `56917701` | 45.64139 | 35.88845 |
| 4 | Five-draw mean of the winning recipe | `56917712` | 45.99039 | 35.51707 |
| 5 | Fresh single draw of the winning recipe | `56917730` | 46.81365 | 40.51026 |

**Stopped: five of five daily slots used; 46 lifetime submissions, all complete. Best remains 30.22376** (`56778295`). The final check was at **17:39:31 UTC**. The queue and predictions never changed after public feedback.

**The winner's 30.22 was a lucky draw.** A fresh single draw of the identical recipe scores **40.51** publicly. The five-draw mean, which includes the 30.22 draw, scores **35.52**. Single draws of one recipe therefore span at least 30.2–40.5 on these three soils.

**Matched resolution is the strongest averaged candidate so far.** Its two-draw mean scores **32.70**, 2.82 better than the five-draw base mean. It changes only test queries, which are downscaled to the examples' ~460 px. Exposure alone scored 35.89. Combined with matched resolution, exposure scored 39.98, so exposure scaling does not appear to help. With two draws per candidate and three public soils, these differences are within plausible noise.

For final selection, `matched_mean` gives up 2.48 public EMD against `56778295`. Under `0.30·public + 0.70·private` it needs private EMD only about **1.06 lower** to break even. Local validation cannot measure its test-only change. Unlike the winner, it is a two-draw average, so less of its score is a single draw's luck.

## First run rejected, amended rerun

The first run stopped all eight recipes as ineligible after 32 dispatches and about 0.64M input tokens. Since October 6, Codex CLI 0.154.0 emits a notice about managed account settings it does not recognize: "Ignoring unknown `features` requirement `ultrafast_mode` …". The declared audit rejects every unrecognized client item, so every attempt was rejected regardless of its answer. Re-validating the preserved attempts showed:

- 30 were otherwise valid;
- 2 were genuine `Selected model is at capacity` failures.

The [amendment](october7-plan.md#amendment-before-the-rerun-1715-utc) accepts only that notice and leaves every other check unchanged. It was committed as `2492581` at **17:03:34 UTC**; its heading's "17:15" is wrong. The first rerun request followed at 17:04:01. The first run's attempts are preserved in `artifacts/experiments/october7/` and never reused. The rerun's eight recipes ran four at a time in `october7r/`: **272 dispatches, all valid on the first attempt**, each carrying two of the accepted notices.

## Local and test-side evidence

Local whole-soil EMD barely measures these changes. Matched resolution affects only test queries and H374. Exposure gains move training images by about 2–3% but iPhone images by about 17–22%. All recipes fall within the base recipe's draw range (44.4–47.4).

| Candidate | Local LOO EMD ↓ | Test distance from the 30.22 winner | Mean shift at 0.063 mm (pp) |
| --- | ---: | ---: | ---: |
| Base redraws D / E | 46.814 / 45.306 | — | — |
| Matched draws A / B | 46.454 / 47.765 | — | — |
| Exposure draws A / B | 47.359 / 44.177 | — | — |
| Aligned draws A / B | 48.529 / 47.266 | — | — |
| **`aligned_mean`** | 47.836 | 14.8 | **+6.4** |
| **`matched_mean`** | 46.772 | 7.9 | +2.5 |
| **`exposure_mean`** | 45.641 | 6.9 | +1.8 |
| **`all_views_five_draw_mean`** | 45.990 | 5.3 | +0.1 |

Rendering test queries like the examples makes the model predict more fine material for test soils. Underestimating fines is its largest systematic error on training soils, at −7 to −9 pp between 0.063 and 0.2 mm. Whether that helps the hidden soils is unknown.

## Frozen queue

The order is the declared fixed order. No candidate was dropped: all are eligible, below 50.62293, and numerically distinct from 41 earlier uploads and each other. Every candidate was produced by `2492581`, whose CI passed. Selection SHA-256: `a8f27bdba6451a15a078485bb7155f575c3c0279b41004b5f5910bfc053550db`; an auditor predicted it independently. Script SHA-256s: `submit.py` `239efae8…125736`, `poll.py` `ac26a6d5…36d668`.

| Order | Candidate | Submission SHA-256 |
| ---: | --- | --- |
| 1 | `aligned_mean` | `3fb68839f954098840bcd731d54402e39bd0dea712658d022332bd8e2c697958` |
| 2 | `matched_mean` | `deecbc72a027ee03c543ecc75e217d7e031b8755ca3320096c1604d63600de07` |
| 3 | `exposure_mean` | `2890053a0b2f5743dddff560b59697af988f243f489f6972cf368f5964f5b867` |
| 4 | `all_views_five_draw_mean` | `fc44aab5a926da5fa3c746a7e413ca5f5a6bf18ee113e2d3834a882e54dd4831` |
| 5 | `all_views_d_vlm` | `1f60172d94582381745848f5f4f78e8cd13d503ea82d3fd900f308b1063b844d` |

## Verification

**663 tests pass**; the runner tests passed before the rerun launched. One unreproduced full-suite failure occurred earlier while review agents ran tests concurrently. Twenty-five isolated runner-test runs and three full reruns passed.

Independent review before launch confirmed the code. It corrected two overstated plan claims before commit: test brightness is not above every training soil (H374 is 157.1), and exposure gains also rescale examples.

The pre-upload audit checked:

- isolation of all 272 attempts;
- that the only non-standard client items were the declared notices;
- byte-identical re-renders of all 3,472 images;
- that the base redraws reproduce the winner's requests;
- that no first-run attempt is reused;
- every score and mean;
- distinctness and the freeze/upload logic.

All passed. GitHub briefly rejected pushes with server errors at about 16:57 UTC; a retry succeeded at 17:01. The same caveats apply: three public soils, mutable hosted aliases, unverified prize eligibility.
