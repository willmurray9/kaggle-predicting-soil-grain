# September 18 execution declaration

The user authorizes autonomous experiments and submissions until an official public EMD beats **55.78511**, or today's allowance is exhausted. This supersedes the September 17 plan's two-upload cap. Fresh authenticated preflight reports **five slots available** and the same current best. Continue to preserve whole-soil folds, reproducible candidates and a clean remote.

## Initial experiments

1. Execute the [camera-transfer diagnostic](superpowers/plans/2026-09-17-domain-and-representation.md): 21 paired soils, 20 training soils per fold, both original Motorola/Samsung directions and matched controls; fixed RGB13 ridge, spectral23 ridge and DINO PLS1. It is diagnostic, not a submission.
2. Test Weibull representation capacity, then spectral23 → two log-parameters ridge if it passes. Fit each label independently with scale bounds **1e-5–1e4 mm**, shape bounds **0.05–20**, Cartesian starting scales **0.002, 0.02, 0.2, 2, 20, 200 mm** and shapes **0.25, 1, 4**. Use SciPy `least_squares(method='trf', max_nfev=2000, ftol=xtol=gtol=1e-10)` and residuals `sqrt(log10 interval width) * (predicted CDF - label)` over the first ten supports. Select the lowest finite converged squared cost, retaining start order on ties; fail explicitly if none converge. Set final support to 100. Capacity gate remains mean reconstruction EMD ≤5 and maximum ≤15. Ridge alpha stays10; no real-label fits before this declaration and producing code are committed.
3. Run a fixed visual-language baseline if isolation works. A synthetic red-image smoke test succeeded through the already authenticated Codex CLI. Use **gpt-6-astra, low reasoning**, fresh ephemeral contexts, user configuration/rules and project instructions disabled, and tools/apps/plugins/memory disabled. No new service purchase. Whole-soil 24-holdout validation uses eight seed-0-selected training examples and all available query camera views, without query labels or soil names. Final inference uses the same rule on all24 labels. One prompt only, recorded before model scoring.

## Visual-language resource and isolation boundaries

Cap at **36 requests including the completed smoke test**, **1,000,000 total input tokens**, and **50,000 total output tokens including reasoning**. Check usage after each request; do not dispatch beyond a consumed cap. Run sequentially so budgets cannot be overshot by concurrent requests. The existing included account capacity is available; no reset, credit purchase or separate paid API is authorized by this implementation.

The CLI does not expose a verified per-request output-token limit. Request only the eleven-number JSON vector, validate its schema and impose a 512-token-equivalent final-text ceiling (2,048 characters); keep the separate aggregate usage ceiling. This is a documented implementation limitation relative to the original plan, not a claim that server-side decoding is capped at512 tokens. Record actual reasoning/output usage and reject tool-use events. Record the model alias and client version; a provider-pinned model revision is not available through this interface, limiting exact replay.

Use one deterministic first photo per training example, and one first photo per available query camera, chosen by sorted camera/path order. Extract a calibrated 100mm center crop using the existing native-to-downloaded pixel scaling, preserve its source pixels up to a768-pixel side, and strip metadata. Do not enlarge small source crops. Present anonymous example/query images in exact prompt order. The fixed preprocessing deliberately preserves more texture than the current256-pixel feature pipeline; this assay tests the combined visual-language representation, not a controlled encoder-only swap.

All24 holdouts must have valid outputs to rank or submit this family. Store raw responses, prompt/image hashes, exemplar IDs outside the model context, actual usage and validation results. Do not retry malformed predictions, revise prompts from errors, drop failed soils, or use public feedback to edit curves. Ten valid test rows are required for upload.

## Follow-up scope and submission selection

If neither initial family wins, use the review's next branches rather than another encoder/blend grid: square-root mass targets, preserved-resolution patch distributions, or an objective matched to absolute CDF error. Declare exact recipes before their evaluation. The camera diagnostic guides which domain hypothesis is useful; it does not reveal the public test IDs. Keep all soils and do not hand-edit labels.

Before the first upload from a batch, freeze the candidates and their upload order using local evidence, including whole-soil EMD, camera-transfer results where available, sensitivity to individual soils, and the specific test-domain hypothesis. Mixed local results can justify informative uploads. Exclude duplicate predictions and invalid candidates. An unsuccessful public result does not authorize adjusting that candidate's predictions against the three public cases.

Each upload requires reviewed tested code, a producing commit pushed with successful CI, valid template/order/ranges, verified source/output hashes, and a receipt saved before the request. Resolve the exact accepted reference before another upload. Stop immediately after a new best or confirmed quota exhaustion, then document results, merge to `main`, push using `will.murray2@icloud.com`, and delete fully merged task branches.

## Progress

- Initial source snapshot: 243 existing artifacts preserved in `/tmp/soilgrain-before-september18.json`.
- Implementation uses a clean task branch with disjoint worker file ownership; the root agent alone handles shared CLI/Makefile integration and Git operations.
- Camera and Weibull implementation are in progress; no model experiment or Kaggle upload has run at declaration time.
