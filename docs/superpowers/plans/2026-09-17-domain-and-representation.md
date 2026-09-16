# Camera Transfer and Curve Representation Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` at implementation time, respecting the user's request for simple, bounded work. Checkboxes track the experiment stages.

**Goal:** Find a defensible improvement over public EMD **55.78511** by addressing validation mismatch and testing distinct representations.

**Architecture:** Reuse verified per-photo caches and existing whole-soil evaluation. Add one camera-transfer diagnostic and one compact curve predictor; assess an isolated visual-language baseline as a separate experiment. Avoid a general experiment framework or another model grid.

**Tech stack:** Existing Python, NumPy, pandas, scikit-learn, Pillow and pytest; SciPy is already available through scikit-learn. A visual-language experiment is conditional on suitable model access and fresh-context execution.

**Spec:** [September 16 strategy review](../../strategy-review-2026-09-16.md).

**Status:** Planned only. Start when the user requests implementation. No scheduled task or automatic submission is created by this plan.

## Global constraints

- Twenty-four labeled soils, not 127 independent training examples. Exclude every photo of a held-out soil from all training and tuning.
- Score the original eleven CDF percentages with `soilgrain.metrics.emd_score`; preserve template column spelling/order and final 100%.
- Learn scaling, target statistics, dimensionality reduction and any selected parameters only within the relevant training fold.
- Keep the public leaderboard out of prompt/weight/hyperparameter tuning. Record candidate recipes and intended upload order before the first new public result.
- Maximum **two uploads in this session**, also bounded by the actual remaining daily quota. Stop after a new best. No upload is mandatory.
- Preserve all earlier artifacts; use separate output directories. Keep personal Git identity `Will Murray <will.murray2@icloud.com>`, tested milestone commits, final merge to `main`, and deletion of fully merged task branches.

## Task 1: Measure transfer between cameras

**Files:** Create `src/soilgrain/camera_transfer.py` and `tests/test_camera_transfer.py`; add one CLI/Make target following existing patterns. Reuse `search_inputs.py`, `ridge.py`, `pls.py`, `metrics.py` and the saved per-photo features. Write new outputs under `artifacts/experiments/camera_transfer/` and a short `docs/camera-transfer-results.md`.

- [ ] Refresh official quota, dataset listing and rules. Compare against September 16; investigate changed files only. Snapshot existing artifacts and verify the input manifests. Do not rerun all historical experiments.
- [ ] Use the **21 soils with both original Motorola and Samsung views**. For each held-out soil, train on exactly the other 20 paired soils. Run Motorola → Samsung and Samsung → Motorola, plus Motorola → Motorola and Samsung → Samsung controls with the same training IDs. Do not include the held-out soil's other-camera photos in training.
- [ ] Evaluate three frozen recipes: original RGB13 ridge (`alpha=10`), spectral23 ridge (`alpha=10`), and DINO PLS (`n_components=1`). Verify the RGB feature mapping against the original cache rather than assuming column order. Average features within each soil/camera and give training soils equal weight. No parameter search.
- [ ] Save every held-out prediction, direction, train/query camera, training soil IDs and source hashes. Report macro-soil EMD in each direction, paired transfer-minus-control differences, per-soil results, and contributions at the fine thresholds. Show the existing pooled 24-soil LOO separately; do not compare their means as if they used identical folds.
- [ ] Add meaningful tests proving exclusion of both views of the held-out soil, exact pairing of transfer/control training IDs, independence from query/test rows, and macro-soil weighting despite unequal photo counts. Run focused tests, review, then commit the diagnostic milestone.

**Decision:** A large transfer penalty supports prioritizing acquisition-robust features. A small penalty weakens this particular camera explanation; it does not establish transfer to unseen iPhones or German soils. This diagnostic cannot identify the three public test IDs and should not attempt to.

## Task 2: Test a compact Weibull target

**Files:** Create `src/soilgrain/weibull.py`, `src/soilgrain/weibull_experiment.py`, focused tests in `tests/test_weibull.py` and `tests/test_weibull_experiment.py`, and one CLI/Make target. Reuse existing spectral cache loading and submission validation. Write only under `artifacts/experiments/weibull/` plus `docs/weibull-results.md`.

The hypothesis is that predicting a typical size and a spread needs fewer examples than predicting ten separate CDF heights. This is inspired by [GRAI3](https://doi.org/10.1016/j.compgeo.2026.108362), not a reproduction of its trained network.

- [ ] Implement `100 * (1 - exp(-(diameter / scale)**shape))` with positive scale/shape represented in log space; output 100 at the final support. Fit parameters to each training curve independently using a deterministic bounded numerical fit with log-width-weighted squared CDF residuals on the first ten supports. Declare numerical bounds, starts and convergence handling in the implementation commit before running real-label fits. Do not tune them against image-prediction scores.
- [ ] First report **label-reconstruction EMD**, without image features or regression. This is approximation error, not an out-of-fold predictive score or a certified global optimum. Predeclare a pragmatic capacity gate: mean reconstruction EMD ≤ **5**, and no soil above **15**. If it fails, document which shapes are poorly represented and stop this family; do not add a mixture of distributions to rescue it.
- [ ] If the gate passes, fit one standardized spectral23 ridge (`alpha=10`, unpenalized intercept) to the two log parameters. Each fold uses only its training soils' parameter targets. Decode to CDFs and score against the original held-out labels. Do not use `ridge_curves` for a two-column target: its output repair assumes eleven CDF columns. Use a small dedicated fit, with no unrelated ridge refactor.
- [ ] Run all 24 whole-soil holdouts and the same camera-transfer/control folds from Task 1. Report all per-soil changes, median change, fine-threshold error and mean change excluding the largest beneficiary as sensitivity diagnostics. Keep every soil in primary scoring and training. Final candidate fits all 24 labels.
- [ ] Test synthetic known Weibull recovery, extreme/flat cumulative curves, finite monotone 0–100 predictions, the exact final support, and independence of a held-out prediction from changes to that held-out label. Test failed fits as explicit failures, not silent mean-curve substitutions. Run focused tests and required suite, review, and commit before any upload.

**Limitation:** The representation-capacity gate uses our known labels and is part of exploratory model selection. It does not make subsequent LOO an untouched confirmation test. Parameter squared loss is also not the competition EMD; all reported predictive scores must use the latter.

## Task 3: Assess one blind visual-language baseline

**Files if feasible:** A small standalone experiment runner and focused context-isolation/output-validation tests, following the existing CLI pattern; record prompts, model/version, image preparation, responses, usage and outputs under `artifacts/experiments/visual_language/`. Document results or the reason for skipping in `docs/visual-language-results.md`.

- [ ] Check existing model access, image limits, reproducible versioning, cost and competition eligibility. Use existing approved capacity only; do not purchase a new service or assume interactive conversation history is blind. Cap the assay at **36 requests**: up to two synthetic schema checks, 24 holdouts and ten test soils. Retries count toward the cap. Freeze model/version and decoding settings before scoring: temperature 0 where supported and at most 512 output tokens per request. Record an input-token/dollar ceiling within existing authorized capacity before sending images; if access, isolation or budget cannot support the full assay, skip it.
- [ ] Lock one prompt before scoring: estimate the eleven **mass percentages passing** from calibrated photos, using training-only worked examples. Use a deterministic seed-0 ordering to choose eight available training soils per fold, with one fixed representative photo each. Remove soil IDs from presented images/metadata, retain useful physical scale, and preserve one fixed image-preparation recipe. Query views come only from the held-out soil. Record the exact prompt and image inputs.
- [ ] Run 24 separate whole-soil holdouts in fresh contexts. Each request receives only its allowed training examples/labels and the query photos, never the query label, prior fold answers, our full conversation, test labels or leaderboard feedback. A coordinating process scores returned predictions afterward. Do not revise the prompt after seeing those scores.
- [ ] Record raw responses and apply one declared deterministic schema/curve-validation policy. Require valid predictions for **all 24 holdouts** before ranking or selecting this candidate; a successful-cases-only mean is not comparable to our baseline. Report failures and costs, with no prompt revisions or repeated answer sampling. Keep this assay to pooled whole-soil LOO; the additional 84 camera-transfer/control requests are outside its budget and scope.
- [ ] Report comparison with spectral ridge, including per-soil and fine-threshold errors, and explicitly note the missing VLM camera-transfer evaluation. If selected for test inference, use the same eight-example rule drawn from all 24 training soils and require all ten valid outputs. This is a new candidate family, not a leaderboard-tuned conversational estimate.

**Motivation and limit:** A participant's reported VLM results are an unverified lead. General visual priors may help infer soil type/fines, but visual surface evidence is not a direct measurement of bulk grain mass. Pretraining overlap with public data may also be unknowable.

## Task 4: Choose submissions and preserve the work

- [ ] Before the first upload, write a short selection note covering both families and their fixed order. A candidate should show a coherent gain in whole-soil accuracy or camera-transfer accuracy without a large unexplained regression in the other. Report whether the gain depends on one soil. The old camera-disagreement gate alone is insufficient; do not optimize a new composite score across these diagnostics.
- [ ] Allow at most one upload per family and **two total**. A mixed local result may justify one informative upload if its transfer evidence and test-domain hypothesis are explicit. No forced fallback, blind blend, public-ID probing, or extra submission just to use quota. If neither family supplies useful evidence, submit neither.
- [ ] Use the existing review, test, template, source-hash and exact submission-receipt safeguards. Wait for each accepted submission's exact reference to complete. Record public scores as three-case feedback; do not change the locked second candidate in response. Stop if the public best is beaten or the session cap/quota is reached.
- [ ] Save results and update the roadmap. Merge tested work to `main`, push with the personal identity, and delete only fully merged task branches. Record any unfinished experiment explicitly for a later session.

## Subsequent branches, not part of tomorrow's sweep

If the compact curve is too restrictive, test a fixed **square-root mass** target next. If image information is the issue, preserve native-resolution local texture and within-soil patch variability before trying another encoder. Verify whether the original archive supplies higher-resolution versions of known training images; deduplicate by physical soil and never count views as new labels. Resolve access, licensing and test overlap before using expanded external data or specialized checkpoints. Consider limited fine-tuning only after these cheaper tests identify a useful representation; do not train a large network from scratch on 24 soils.
