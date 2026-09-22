# September 22 transport-retry deviation

Recorded during inference, before either recipe has a complete local evaluation or any new Kaggle result. The declaration and producing code remain unchanged.

The existing Codex CLI performs internal transport retries that the runner did not disable. This **deviates from the declaration's literal blanket prohibition on automatic retries**. The runner itself has not relaunched any requested, failed or uncertain CLI invocation. Request counts in its ledger count CLI dispatches, not all underlying server attempts.

The `all_examples_vlm` recipe's second dispatch, held-out G190, logged five sampling retries, an HTTP fallback, four reconnect error events and then one completed turn. The original, precommitted event validator rejects any error event. The recipe therefore remains failed and ineligible: do not repair or reuse that response, resume it, or compute a candidate from incomplete holdouts. Use the declared saved replacement if the other recipe completes.

The multi-scale H030 dispatch logged one transport retry in stderr but no error event and exactly one valid completed turn. Continue that recipe under the same precommitted event validator. It is eligible only if all 34 responses pass the existing checks. This is an explicitly disclosed infrastructure deviation, not literal compliance with the original no-retry wording. The independent review found no prediction selection, prompt change, label leakage or public-feedback-dependent decision in retaining otherwise valid responses. The user's broad autonomous authorization covers continuing the unchanged experiment; no additional recipe or model request retry is introduced.

Reported resource usage has a related limitation. G190's completed event reports **23,544 input / 295 output tokens**, including **236 reasoning tokens**, but these are absent from its failed ledger entry because event validation raises before returning usage. Count that known completed-event usage separately in the final audit without modifying the failed artifacts. Usage of disconnected server attempts is not established by the available logs. Do not claim that recorded totals prove a strict bound on all provider-side consumption, or that there were no transport retries. The declared aggregate checks still apply to recorded usage; no service capacity was purchased or reset.

Preserve the failed recipe, raw logs and all earlier artifacts. No new public score has been inspected, and no candidate recipe, curve, upload priority or failure replacement changes in response to this finding.
