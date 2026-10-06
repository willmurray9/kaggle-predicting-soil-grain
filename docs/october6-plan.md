# October 6 experiment declaration

The user asked to keep trying to beat the public best of **30.22376** (`56778295`, the single all-photo GPT draw) or use today's five slots. Preflight at **15:39 UTC** found 36 completed submissions, none today, five slots, a clean `main` and green CI. The user declined leaderboard probing. Codex CLI is unchanged (0.154.0). The Claude Code build used on October 2 (2.1.286) has been replaced. Claude scored worst publicly (42.00414), so this round uses GPT only.

## What the evidence says

- **Expectations.** The three all-photo 100 mm GPT draws averaged **33.38** publicly. By convexity, the two fresh draws averaged at least about 35 on their own. So 30.22 includes draw luck, and this recipe family's public expectation is roughly 33–35. Every candidate below is one the user could reasonably select as a final submission, since 70% of the score is private.
- **Example sets (panels).** Every past change to the examples lost publicly: retrieval 47.3, paired cameras 46.0, all 23 examples 58.5.
- **Local diagnostics without model calls.** The all-photo model underestimates fines by 7–9 points at 0.063–0.2 mm. A simple learned per-support correction worsens local error (50.50 vs 46.11), because the bias varies by soil. GPT/Claude means gain under 2 points.

## Hypotheses

1. **Bagging over example panels.** Every test soil shares one seed-0 panel, mostly gravels: H615, H038, H637, H368, H371, H030, H666, H181. The random draw always permutes 23 IDs for a holdout, so it picks the same eight positions each time. Holdouts therefore saw only nine near-identical seed-0 panels: two used exactly the test panel and ten more shared seven of its eight soils. Local seed-0 error mostly measures that one panel family. If it suits the test soils poorly, every prediction suffers together. Averaging over panels reduces dependence on any one panel.
   - Each panel recipe's local error describes its own holdout-panel family, which overlaps its test panel only partly (one to eight soils). It cannot rank individual test panels.
   - Seeds 1–4 change only the example lines; their test panels include more fines-rich soils.
   - The panel mean keeps random eight-example panels, unlike the full-context recipe that lost publicly.
2. **More coverage at the winning scale.** Showing every photo helped locally and publicly. The round-2 150 mm recipe widened the view for examples and queries alike and lost publicly. Tiles instead keep every image at the winning 100 mm while adding coverage. Each query photo contributes **two 100 mm tiles** centered ±50 mm from its center along the long side, covering 200 × 100 mm. Examples stay as original center crops.
   - A screen of all 324 tiles found only soil. The Münster ruler starts 59 mm below center, outside the ±50 mm short-side extent.
   - The crop sentence becomes "Each attached image is a 100 mm square crop."
   - The query count states the number of tiles.

## Recipes

All use gpt-6-astra at low reasoning, all query photos sorted by camera/path, and 100 mm crops capped at 768 px. They inherit the October 2 isolation, budgets, two-dispatch validity rule and transport policy. Outputs go to `artifacts/experiments/october6/`. The new code reproduces every October 2 request byte-for-byte; the seed-0 prompt equals `build_prompt` exactly.

| Recipe | Example panel | Query images |
| --- | --- | --- |
| `panel{1,2,3,4}_all_views_vlm` | `default_rng(seed).permutation(sorted other training IDs)[:8]` | center crops |
| `tiled_all_views_{a,b}_vlm` | seed 0 (original) | two ±50 mm tiles per photo |

Launches are staggered by a few seconds to avoid the Codex start-up race seen on October 2.

## Candidates, queue and stopping

Each candidate is an equal-weight coordinatewise mean of its components, aligned by soil ID; there are no fitted weights. A component counts only if it finished eligible.

| Candidate | Components (equal weight) | Minimum eligible |
| --- | --- | ---: |
| `panel_bagged_mean` | saved `all_views_draw_mean` (seed 0, three draws) + panels 1–4 | 4 of 5 |
| `tiled_mean` | the two tiled draws | 2 of 2 |
| `coverage_panel_mean` | `panel_bagged_mean` + `tiled_mean` | 2 of 2 |

The queue is:

1. **Slots 1–3:** the three means, by whole-soil LOO EMD ascending.
2. **Slots 4–5:** the two tiled single draws, by LOO. Like the uploaded winner, they use the seed-0 panel, so their local error is relevant to their test panel.

Ties break by name. A candidate is dropped if it is ineligible, numerically duplicates any of the 36 earlier uploads or another queued file, or has LOO EMD above **50.62293**. Vacancies fill from single panel draws in seed order (1–4), since their local error cannot rank test panels. The CSV hashes and order are frozen before any upload. Round scripts live in `artifacts/experiments/october6/`:

- target 30.22376;
- 36 known references;
- the date guard set to 2026-10-06;
- stop at the first public score below 30.22376 or when the allowance is used;
- no changes after public feedback, and no inference about which soils are public.

## Limits

Expect about 4.5M Codex input tokens; no capacity purchase or reset. Local LOO is descriptive apart from the drop and ordering rules. Public EMD covers three soils and cannot establish private performance. Hosted aliases are not immutable, and external-model prize eligibility remains unverified.
