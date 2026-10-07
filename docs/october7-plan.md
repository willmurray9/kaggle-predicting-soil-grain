# October 7 experiment declaration

The user asked to keep going: beat the public best of **30.22376** (`56778295`) or use today's five slots. Preflight at **16:37 UTC** found 41 completed submissions, none today, a clean `main`, green CI and Codex CLI 0.154.0. The user declined leaderboard probing.

## Evidence

The all-photo GPT family has nine public scores. Seven fall between 33.4 and 39.4, the 150 mm mean scored 41.2, and the single best draw scored 30.22. Every change to the example panels or crop geometry has scored worse publicly than the seed-0, 100 mm, all-photo center-crop recipe. This round keeps that recipe and changes only how images are rendered. Matched resolution changes only query resolution: the ten test queries, plus H374's holdout query. Exposure scaling applies one per-camera gain to every image, examples and training queries included. It moves training images by only about 2–3%, but iPhone images by about 17–22%. Both target measured differences between the training and test domains:

1. **Resolution.** Example crops are ~460 px (~4.6 px/mm downloads). iPhone query crops are 768 px, downsampled from 1394–1953 px. Training queries are also ~460 px, except H374 at 768, so whole-soil validation barely measures this change.
2. **Exposure.** Mean luminance of 100 mm center crops (Rec. 601, measured at 256 px):

   | Camera | Mean luminance |
   | --- | ---: |
   | Motorola Edge | 126.0 |
   | Samsung A52 | 132.6 |
   | Motorola Edge 60 Fusion (H374 only) | 157.1 |
   | iPhone 14 | 155.5 |
   | iPhone 16 | 166.2 |

   Brightness varies among training soils: Motorola soil means run from 115.6 to 142.0. It correlates modestly with fines (r = 0.39 Motorola, 0.34 Samsung); fines-rich soils are about 8 levels brighter than gravelly ones. But the ten test soils' means run from 146.9 to 178.9, above every Motorola Edge and Samsung soil (max 142.0), for gravels and sands alike. The only brighter training soil is H374 (157.1), the single gravelly soil photographed on the brighter Motorola Edge 60 Fusion, which also points to a camera effect. One Münster photo (130.0) falls inside the training range. The gap is mostly between domains (camera, lighting), so per-camera rather than per-image normalization preserves within-camera soil differences.

## Recipes

All use gpt-6-astra at low reasoning, the original prompt text unchanged, the seed-0 eight-example panel, and all query photos sorted by camera/path as 100 mm center crops. They inherit the October 2 isolation, budgets, two-dispatch validity rule and transport policy. Outputs go to `artifacts/experiments/october7/`.

| Recipe (two draws each) | Crop cap | Exposure |
| --- | ---: | --- |
| `all_views_{d,e}_vlm` (base redraws) | 768 | unchanged |
| `matched_all_views_{a,b}_vlm` | **460** | unchanged |
| `exposure_all_views_{a,b}_vlm` | 768 | **per-camera gain** |
| `aligned_all_views_{a,b}_vlm` | **460** | **per-camera gain** |

**Exposure gains** map each camera's mean crop luminance to the training mean, **129.615**. They were computed once, before any request, from unlabelled 100 mm center crops of every photo; test photos contribute only pixels, no labels. They are frozen in code:

| Camera | Gain |
| --- | ---: |
| Motorola Edge | 1.0283 |
| Motorola Edge 60 Fusion | 0.8248 |
| Samsung A52 | 0.9776 |
| iPhone 14 | 0.8333 |
| iPhone 16 | 0.7799 |

The Motorola Edge 60 Fusion photographed one soil, so its gain is confounded with that soil. The iPhone gains are confounded with the test soils for the same reason, but the soil-related brightness effect within training is far smaller than the domain gap. The gain is applied after resizing to every image, examples included: each channel value is multiplied by the gain, rounded half to even, and clipped to 0–255.

The base redraws must rebuild the uploaded winner's prompts and images byte-for-byte before any request. Without a gain, `prepare_crop` is byte-identical to before. Launches are staggered.

## Candidates, queue and stopping

Each candidate is an equal-weight coordinatewise mean of eligible components, aligned by soil ID.

| Order | Candidate | Components | Minimum |
| ---: | --- | --- | ---: |
| 1 | `aligned_mean` | aligned a, b | 2 |
| 2 | `matched_mean` | matched a, b | 2 |
| 3 | `exposure_mean` | exposure a, b | 2 |
| 4 | `all_views_five_draw_mean` | saved all-photo draws A, B, C + d, e | 4 |
| 5 | `all_views_d_vlm` | one fresh base draw | — |

The test-domain alignments go first. The five-draw mean is the lowest-variance version of the best public recipe. Slot 5 measures how typical a fresh single draw of the winner is. If it beats 30.22376, the new best is again a single draw on three soils, and we will say so.

A candidate is dropped if it is ineligible, numerically duplicates any of the 41 earlier uploads or another queued file, or has LOO EMD above 50.62293. For these mostly test-side changes that threshold is nearly vacuous; local LOO for the exposure recipes reflects only the small training-side rescaling. Vacancies fill in order from `all_views_e_vlm`, then the single aligned, matched and exposure draws (a before b).

Round scripts in `artifacts/experiments/october7/`:

- target 30.22376;
- history of 41 references (the October 6 selection plus its five receipts);
- distinctness checked against 41 CSVs;
- date guard 2026-10-07;
- stop at the first public score below the target or when the allowance is used;
- no changes after public feedback, and no inference about which soils are public.

## Limits

Expect about 5M Codex input tokens; no capacity purchase or reset. Local LOO cannot measure the test-domain effect of these changes, so public scores on three soils are the only, very noisy, test-domain signal. They cannot establish private performance. Hosted aliases are mutable, and external-model prize eligibility remains unverified.

## Amendment before the rerun (17:15 UTC)

The first run was preserved in `artifacts/experiments/october7/` and is never reused. All eight recipes stopped as ineligible, after 32 dispatches and about 0.6M input tokens.

**Why it failed.** Since October 6, Codex CLI 0.154.0 emits a notice before every answer: "Ignoring unknown `features` requirement `ultrafast_mode` from requirements layers: enterprise-managed requirements …". The account's managed settings gained a feature this client version does not recognize. The declared audit rejects any unrecognized client item, so every attempt was rejected regardless of its answer.

Re-validating the preserved attempts with the amended rule shows:

- 30 were otherwise valid;
- 2 were genuine `Selected model is at capacity` turn failures, which the amended validator still rejects.

**The amendment** applies only to the October 7 recipes:

- An `item.completed` error item is accepted only if it fully matches `Ignoring unknown \`features\` requirement \`[a-z_]+\` from requirements layers: enterprise-managed requirements .+`. This parallels the long-accepted "Code Mode is unavailable" notice. The number of notices is recorded.
- All other checks are unchanged: one completed turn, one final message matching the response, no tool use, and the September 23 transport and stderr rules.
- The same eight recipes rerun as fresh requests in `artifacts/experiments/october7r/`, under the same budgets and two-dispatch rule.
- To reduce capacity failures, four recipes run at a time: base redraws and aligned first, then matched and exposure.
- The candidates, queue, drop rule and round scripts are unchanged except that they read `october7r/`.
