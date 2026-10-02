# October 2 experiment declaration

The user asked for a fresh review, new ideas, and autonomous work until a confirmed public EMD beats **37.62858** or today's five-submission allowance is used. Official preflight at **15:57 UTC** found 31 completed submissions, none today, five slots, and a November 30 deadline. Preserve all historical artifacts; new outputs belong under `artifacts/experiments/october2/`.

## Why these experiments

Every visual-language (VLM) change that altered the original context has lost publicly, so this round stays close to the winning recipe. Each candidate changes one thing. Three target concrete mismatches between how training and test soils were shown to the model:

1. **Test soils saw one photo.** The original takes one photo per camera. Training soils have two cameras, so their holdouts saw two views. Every test soil has a single iPhone camera, so it saw **one of its 3–5 photos**. That covers just 100 mm of soil, too little for the cobble and gravel test soils.
2. **Test crops are sharper than the examples.** The labelled examples are ~460 px crops (~4.6 px/mm downloaded resolution). iPhone query crops are downsampled from 1394–1953 px to 768 px. The model therefore compares examples and queries at different resolutions.
3. **Single draws are noisy.** Hosted responses vary from run to run. EMD is convex in the prediction, so averaging independent draws of one recipe has expected EMD no worse than one draw.

The other two ask whether more deliberate reasoning, or a different model family, reads the same evidence better.

## Recipes

All recipes keep the original prompt text (`visual_language.build_prompt`), its eight seed-0 training examples, and anonymous request folders. They also keep the 100 mm calibrated center crops, sorted camera/path photo order, and the eleven-value JSON schema. The held-out soil's photos and label never appear in its examples.

| Recipe | Model / reasoning | Query photos | Crop side cap |
| --- | --- | --- | ---: |
| `rerun_a_vlm`, `rerun_b_vlm` | gpt-6-astra / low | first per camera (original) | 768 |
| `all_views_vlm` | gpt-6-astra / low | **all photos of the query soil** | 768 |
| `matched_resolution_vlm` | gpt-6-astra / low | first per camera | **460** |
| `high_reasoning_vlm` | gpt-6-astra / **high** | first per camera | 768 |
| `claude_vlm` | **claude-opus-5-5** / low effort | first per camera | 768 |

The reruns must reproduce the original requests' prompt and image SHA-256 values exactly. They are checked against `artifacts/experiments/visual_language/requests.json` before any request.

**Codex** uses CLI 0.154.0 with the original flags: ephemeral, read-only sandbox, user config/rules ignored, and tools, apps, plugins, hooks and memories disabled. Only the reasoning setting differs for `high_reasoning_vlm`.

**Claude** uses the desktop-bundled Claude Code 2.1.286 in print mode with `--safe-mode`, `--tools ""` and strict empty MCP. It also loads no setting sources or slash commands, skips session persistence, and uses a minimal system prompt and the same JSON schema. Images go in as base64 blocks, in the original order, before the prompt text, from an empty anonymous folder. A blue-square smoke test answered correctly at about 1.2k input tokens: the image arrived and no CLAUDE.md, plugin or tool content leaked in. Accept a response only if all of these hold:

- the init event reports model `claude-opus-5-5`, tools `["StructuredOutput"]` and no MCP servers;
- the only tool use is `StructuredOutput`;
- the result is a success.

**Derived candidate `vlm_draw_mean`:** the coordinatewise mean of the saved original VLM, `rerun_a_vlm` and `rerun_b_vlm`. It is formed for the 24 out-of-fold rows and the ten test rows, with no fitted weights. If exactly one rerun fails, it averages the original and the valid rerun.

## Execution limits and validity

Each recipe makes 34 queries: 24 whole-soil holdouts, then the ten test soils. Each query gets at most **two dispatches**. A second dispatch is allowed only after an objective failure: client error, timeout, transport/audit rejection, or a curve that is malformed or non-monotone. A plausible-looking curve is never re-requested. Every attempt's raw files are preserved, and the first valid attempt is used. A recipe with any query lacking a valid attempt is ineligible. Curves are never repaired or edited.

Budgets per recipe, checked before each dispatch (in-flight requests may finish):

- **Codex at low reasoning:** 2,000,000 input and 60,000 output tokens.
- **`high_reasoning_vlm`:** 2,000,000 input and 600,000 output tokens.
- **Claude:** 2,000,000 input tokens (including cache), 150,000 output tokens, at most $2 per call and $60 per recipe.

Rejected attempts count against these budgets at their reported usage. If usage is unknown (timeout, killed client), the attempt is charged a ceiling: 40,000 input tokens and $2 for Claude. A recipe stops dispatching once any query has used both attempts, because it is already ineligible. A client that cannot start sends no request, so it does not consume an attempt. If an attempt was dispatched before an interruption, its response is audited on resume and used if valid, never re-requested. Timeouts are 300 s for low reasoning and 900 s for high. At most two concurrent requests per Codex recipe and three for Claude. Codex transport recovery follows the September 23 policy: recognized reconnect/fallback messages are accepted only with exactly one completed turn and a final message that matches the response. No capacity purchase or reset.

## Frozen queue and stopping

Primary upload order, fixed now: **`all_views_vlm` → `vlm_draw_mean` → `matched_resolution_vlm` → `high_reasoning_vlm` → `claude_vlm`.** The two test-targeted fixes go first, then the noise reduction closest to the winner, then the larger model changes.

A candidate is dropped and replaced, in order, by `rerun_a_vlm` and then `rerun_b_vlm` if any of these hold:

- it failed;
- its CSV numerically duplicates an earlier upload or queued file;
- its whole-soil LOO EMD exceeds **60.0**, i.e. clearly worse than the original's 50.62293.

Local scores are otherwise descriptive. All CSV hashes and the order are frozen before any upload.

Before each upload, verify:

- the UTC date, quota, current best and known history;
- that the producing commit is pushed with successful CI;
- template order, valid CDFs, and numerical distinctness.

Save a receipt before uploading. Stop at the first public score below **37.62858** or when the allowance is used. No prediction or queue change follows public feedback. We do not infer the public soils from historical scores.

## Limits

Hosted model aliases are not immutable checkpoints, so fresh requests may differ from saved responses. External-model accessibility and prize eligibility remain unverified. The public score covers three soils and does not establish private performance. This session has seen training labels and test images, so all predictions come only from isolated CLI subprocesses; no curve is written by hand.
