# September 22 experiment declaration

The user renews autonomous work until an official public EMD beats the current **37.62858** or the current daily allowance is exhausted. Fresh official preflight confirms twenty-one completed submissions, none today, five available, and unchanged competition pages/data. Keep whole-soil validation, preserve every historical artifact, commit producing code before model requests, and finish with tested work pushed to clean `main` using the personal Git email. Existing autonomous authorization covers these bounded experiments; no paid service, capacity reset or organizer contact is requested.

## Two information hypotheses

1. **`multiscale_vlm`: matched broad and close views.** Keep the original deterministic eight training examples, model, low reasoning and mass-CDF task. Each example photo and query view receives two images: the original calibrated 100 mm center crop followed by a calibrated 25 mm center crop, both extracted directly from the source photo. Each crop preserves its native pixels up to the existing 768-pixel side cap; never enlarge a crop. Label both physical widths and same-photo grouping in the prompt. This exposes finer test-image texture while retaining broad composition. It does not recover unresolved training grains or make visible area equivalent to dry mass.
2. **`all_examples_vlm`: complete labeled context.** Use all other 23 training soils in each holdout and all 24 training soils for test queries, in alphabetical ID order outside the model context. Use one original 100 mm crop per example, and the original one query view per available camera. Keep the model, reasoning and task fixed. This tests whether omitted reference soils limited inference. Do not choose examples from target labels, nearest neighbors or leaderboard feedback.

For both recipes, select the same first example photo by camera/path ordering as the original runner. Query images remain the first photo per available camera. Provide anonymous image names and no soil IDs, hidden query labels, filenames or retrieval tools. Every photo of the held-out soil is excluded from examples. There is no combined multiscale/all-examples candidate, fitted blend, prompt search, image-location search or parameter sweep.

Use the existing authenticated Codex CLI, model alias `gpt-6-astra`, low reasoning, fresh ephemeral contexts, and the original disabled-tool/app/memory/repository-instruction settings. Allow at most **34 requests per recipe (68 total)**, **2,000,000 input tokens per recipe**, and **50,000 total output tokens including reasoning per recipe**. Budgets stop further dispatch and make an over-budget recipe ineligible; these are aggregate limits, not verified server-side per-call caps. Final text remains limited to 2,048 characters. Save a receipt before each request, all prompts/crops/raw events/responses, code/client identity and source hashes. No automatic retries, malformed-curve repair or dropped holdouts. Require all 24 held-out and ten test predictions to be valid. Stop an invalid or unavailable recipe with explicit failure evidence and continue only the other declared recipe or saved replacements.

Compute whole-soil EMD and descriptive per-soil gains against the original VLM, including the mean excluding the largest beneficiary. Do not use these diagnostics to change recipes or impose a post-hoc upload gate. No new camera-transfer claim is made. Local evidence remains exploratory after repeatedly using these 24 soils.

## Fixed candidate pool and submission order

Finish both eligible VLM outputs before reading any new public result. Order the two by ascending local EMD, with experiment name breaking ties. Then use these already evaluated, never-submitted candidates in this order:

1. `native_sqrt_mass_ridge`: native-resolution features and square-root mass targets; local EMD 39.90205.
2. `lbp_ridge`: local binary-pattern texture representation; local EMD 45.57380.
3. `emd_median_regression`: regression aimed at the absolute-error metric; local EMD 48.58551.

If one or both VLM recipes fail to complete, fill the missing slots after those three with the saved `native_lbp_blend` and then `spectral_boost`, taking only as many as necessary. All numerical outputs are reused byte-for-byte with their original producing commits and provenance. No replacement fitting is authorized by this declaration. A candidate that fails integrity, official-rule or numerical-duplicate checks cannot be uploaded; resolve that concrete issue before progressing.

Freeze at most five eligible CSV hashes and the full order before the first upload. Verify current best, current UTC date, daily allowance, exact known submission history, template order/ranges/monotonicity, numerical distinctness, source hashes and successful producing-code CI before each upload. Save request receipts before upload and resolve every accepted or uncertain reference before another. Stop immediately on a confirmed improvement or exhausted allowance; do not adjust remaining predictions from public feedback.

## Preservation and limits

Write only new experiment artifacts under `artifacts/experiments/september22/`. Add a small runner and focused tests without changing historical producing modules. Preserve all prior artifacts and capture an initial hash inventory. Store the selection, receipts, final official state, independent audits and final artifact inventory alongside the outputs. Code and reports belong in Git; data and generated artifacts retain the existing local-only policy. Merge to `main`, push and delete the fully merged task branch.

Saved hosted responses reproduce the exact submitted CSV, but the hosted alias is not an immutable model checkpoint and fresh responses may differ. External-model accessibility and organizer acceptance remain unverified; successful scoring does not establish prize eligibility. The user has already been informed and renewed authorization. The three public soils do not establish private-set performance or resolve pretraining-overlap uncertainty.
