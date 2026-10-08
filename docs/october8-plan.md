# October 8 experiment declaration

The user asked to keep going: beat the public best of **30.22376** (`56778295`) or use today's five slots. Preflight at **15:43 UTC** found 46 completed submissions, none today, a clean `main`, green CI and Codex CLI 0.154.0. The user declined leaderboard probing.

## Why replicate matched resolution

This direction was chosen **after** a public result. On October 7, the matched-resolution two-draw mean scored **32.70180**, best of five candidates on three public soils. The base all-photo recipe's five-draw mean scored **35.51707**. A fresh single draw of that base recipe scored 40.51 against the uploaded draw's 30.22, so single-draw noise is large. Because matched was the best of several noisy candidates, regression to the mean is expected: a fresh replication is more likely to score worse than 32.70 than better. This round tests whether the 32.70 replicates.

The recipe downscales test-query crops to the labelled examples' ~460 px. It changes only test queries and H374's holdout query, so whole-soil validation barely measures it.

## Recipes

Six fresh draws, `matched_all_views_{c,d,e,f,g,h}_vlm`, are identical to October 7's `matched_all_views_a/b_vlm`:

- gpt-6-astra at low reasoning;
- the original prompt and seed-0 example panel;
- all query photos as 100 mm center crops capped at 460 px;
- the October 7 amended validator, which accepts only the managed-requirements notice;
- the inherited isolation, budgets and two-dispatch rule.

Before any request, all 34 prompts and image sets were checked to reproduce October 7's matched requests byte-for-byte. Outputs go to `artifacts/experiments/october8/`. At most three recipes run at a time, staggered, to limit capacity failures. The first live attempts are checked before the rest are trusted.

## Candidates, queue and stopping

Each candidate is an equal-weight coordinatewise mean of its eligible components.

| Order | Candidate | Components | Minimum |
| ---: | --- | --- | ---: |
| 1 | `matched_new_six_mean` | c–h | 5 |
| 2 | `matched_eight_draw_mean` | October 7 a, b + c–h | 6 |
| 3 | `matched_cd_mean` | c, d | 2 |
| 4 | `matched_ef_mean` | e, f | 2 |
| 5 | `matched_gh_mean` | g, h | 2 |

1. **The replication test.** It contains no publicly scored draw.
2. **The intended final-submission candidate.**
3. **Slots 3–5:** two-draw replications, the same size as the 32.70 mean. They mostly add variance, like lottery tickets with a replication label.

A candidate is dropped if it is ineligible, numerically duplicates any of the 46 earlier uploads or another queued file, or has LOO EMD above 50.62293, a nearly vacuous threshold here. Vacancies fill in order from the single new draws c–h.

Round scripts in `artifacts/experiments/october8/`:

- target 30.22376;
- history of 46 references (the October 7 selection plus its five receipts);
- distinctness checked against 46 CSVs;
- date guard 2026-10-08;
- stop at the first public score below the target or when the allowance is used;
- no changes after public feedback, and no inference about which soils are public.

## Interpretation, declared before any result

- **Matched resolution is consistent with a benefit** if and only if `matched_new_six_mean` scores below the base five-draw mean (35.51707). Otherwise the October 7 32.70 is treated as noise. Same-recipe means already differ by about 2 EMD on three soils (33.38 for three draws, 35.52 for five), so one comparison cannot establish a benefit. A small margin either way is weak evidence.
- **The eight-draw mean is not a replication.** Its public score includes draws a and b, which produced the publicly selected 32.70.
- **Final-selection advice:** if matched is consistent with a benefit, recommend `matched_eight_draw_mean` as the final candidate, whatever the two-draw means score. Picking the lowest two-draw mean would be selection on three-soil noise.
- **If any candidate beats 30.22376,** it is reported as another small-sample public result. A two-draw winner especially would be luck on three soils, not established superiority.

## Limits

Expect about 3.8M Codex input tokens; no capacity purchase or reset. The public score covers three soils and cannot establish private performance. Hosted aliases are mutable, and external-model prize eligibility remains unverified.
