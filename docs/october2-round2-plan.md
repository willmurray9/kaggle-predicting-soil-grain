# October 2 second-round declaration

Showing the model every photo of the query soil set a new public best of **30.22376** (`56778295`) with the [first round](october2-results.md)'s first upload. The user declined leaderboard probing. They asked to continue with the next documented step until public EMD beats **30.22376** or today's four remaining slots are used.

## Hypotheses

1. **Averaging draws of the winner.** A fresh draw of an identical recipe differs from a saved one by about 9 EMD per holdout. EMD is convex in the prediction, so a coordinatewise mean of independent draws has expected error no worse than one draw. The winner's 30.22376 is itself a single draw.
2. **Wider crops for coarse soils.** A 100 mm square holds only a few cobbles or large gravel pieces. 150 mm is the widest square center crop that fits every camera: the Motorola Edge short side is 156.6 mm. A screen of all 162 photos at 150 mm found no tray rims, only shadows between stones. Independent review then found one exception: test photo `iPhone14_HPC_Münster_BS6_9,0-10m (5).JPG` (SHA-256 `8d6d82eb…5d79b94`) shows a ruler and a company card at the bottom edge, starting about 59 mm below center. Any uniform width that excludes them is under 116 mm, too small to test this hypothesis. The 150 mm recipes therefore omit that one photo; Münster keeps its other four. The exclusion is fixed by file hash before any request. The 100 mm recipes keep it, because the ruler is outside a 100 mm crop. Examples and queries both use 150 mm. The examples render at ~690 px; query crops are capped at 768 px.
3. **A second model family with all photos.** Claude Opus 5.5 matched the original GPT recipe locally. It is cheap, so three averaged draws reduce its larger single-draw noise.

## New model recipes

Every recipe uses all query photos sorted by camera/path, the seed-0 eight examples, the original prompt, low reasoning and the round-1 isolation, budgets, two-dispatch validity rule and transport policy.

| Recipe | Model | Crop | Draws |
| --- | --- | --- | ---: |
| `all_views_b_vlm`, `all_views_c_vlm` | gpt-6-astra | 100 mm | 2 new |
| `wide_all_views_{a,b,c}_vlm` | gpt-6-astra | **150 mm** | 3 |
| `claude_all_views_{a,b,c}_vlm` | claude-opus-5-5 | 100 mm | 3 |

The 100 mm redraws and Claude draws must rebuild the saved `all_views_vlm` prompts and images byte-for-byte before any request. In the 150 mm recipes, the only prompt change is "100 mm" → "150 mm" in the crop sentence. The 100 mm redraws start first so a usage limit is least likely to affect slot 1.

## Candidates, queue and stopping

Each candidate is the coordinatewise mean of its components' 24 out-of-fold and ten test rows, with no fitted weights. A component counts only if its recipe finished eligible. The candidate is dropped if fewer than the stated minimum are eligible.

| Order | Candidate | Components | Minimum |
| ---: | --- | --- | ---: |
| 1 | `all_views_draw_mean` | saved `all_views_vlm` + `all_views_b/c` | 2 |
| 2 | `all_scales_draw_mean` | the three 100 mm and three 150 mm GPT draws | 4 |
| 3 | `wide_draw_mean` | the three 150 mm draws | 2 |
| 4 | `claude_all_views_mean` | the three Claude draws | 2 |

A candidate is also dropped if it numerically duplicates an earlier upload or queued file, or if its whole-soil LOO EMD exceeds the original VLM's **50.62293**. Dropped candidates are replaced in order by the single draws `wide_all_views_a_vlm` and then `all_views_b_vlm`. The CSV hashes and order are frozen before any upload.

Round-2 upload scripts live in `artifacts/experiments/october2_round2/`:

- target **30.22376**;
- **32 known submissions**, including `56778295`;
- distinctness checked against all 32 earlier files;
- the existing receipt, CI, template and quota checks, and stop at the first public score below 30.22376.

No prediction or queue changes after public feedback, and we do not infer which soils are public.

## Limits

This round adds roughly 4–5M Codex input tokens to the ~3.6M already used today. Claude costs a few dollars. No capacity purchase or reset. Local LOO is descriptive except for the drop rule. Public EMD covers three soils and cannot establish private performance. Hosted aliases are not immutable, and external-model prize eligibility remains unverified.
