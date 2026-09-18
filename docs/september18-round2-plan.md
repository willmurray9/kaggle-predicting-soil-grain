# September 18 second-round declaration

The user renewed autonomous work after public EMD improved to **49.65594**. Stop immediately when an official score beats that incumbent or today's remaining allowance is exhausted. The preceding round used two of five slots; refresh the actual quota before uploading. This is a new experiment batch, not a revision of its frozen predictions or receipts.

## Three fixed hypotheses

1. **Native spectral square-root mass ridge.** Use the winner's existing 23 features from calibrated 100 mm, 460-pixel crops. Apply the already tested square-root mass predictor unchanged: training-only standardization, ridge alpha10, clipping negative roots, squaring, normalizing, and cumulative summation. This tests whether the flexible mass representation's modest local benefit survives at the improved image resolution. Its previous 256-pixel public result was worse; improvement is uncertain.
2. **Native spectral / LBP equal blend.** Average the winner's CDF prediction with the fixed 40-feature local binary pattern ridge CDF, exactly 50/50. Each component independently fits its scaling and alpha10 ridge on the same training soils. There is no weight search. LBP describes brightness-order texture distributions; averaging predictions limits reliance on one representation. Evaluate pooled and camera-transfer folds with both components using identical held-out soils.
3. **Existing isolated visual-language baseline.** Reuse its original 24 held-out and ten test predictions, byte-for-byte. No new requests, prompt revision, test-ID targeting, or example changes. Its local EMD is 50.62293 and transfer to the public soils is unknown. This is the most distinct representation in the batch.

No regressor grid, additional crops, new encoder, or further model family is planned in this round. The unsubmitted standalone LBP and median-regression candidates remain saved; this round does not upload them.

## Evaluation and upload order

Before fitting, commit this declaration. Reuse verified immutable source caches and existing evaluation functions. Evaluate the two numerical candidates with all 24 whole-soil holdouts and all 21 paired-camera folds; compare against the new native-resolution incumbent. Report per-soil gains, sensitivity to the largest beneficiary, and camera transfer. These are exploratory comparisons after repeated use of the same 24 labels.

Run focused tests for composition and alignment, an independent review, and the full suite. Commit and push the producing code before fitting real candidates. Freeze all three CSV hashes and the upload queue before the first new public result. **Upload the visual-language baseline first**, because it tests a different visual prior; then upload the two numerical candidates in ascending local EMD order (name breaks ties), if the goal has not been met. A weaker local result is eligible as an informative fixed hypothesis given the observed local/public mismatch. Do not revise any recipe, weight, or order from this round's public scores.

Before each upload, verify official quota and incumbent, complete all outstanding submission receipts, check producing-commit CI, recheck source/output hashes, reject duplicate predictions, and validate the ten-row template. Record a request receipt before contacting Kaggle; never retry an uncertain request blindly. Preserve every earlier artifact. Save new results under `artifacts/experiments/september18_round2/`, document accepted references and scores, merge and push tested work to `main` with `will.murray2@icloud.com`, and delete the fully merged task branch.
