import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.constants import CANONICAL_GRAIN_LABELS


def inputs(tmp_path):
    train = [f"S{i:02d}" for i in range(24)]
    test = [f"T{i:02d}" for i in range(10)]
    truth = pd.DataFrame([np.arange(0, 101, 10) for _ in train], columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0, "sample_id", train)
    sample = pd.DataFrame([np.zeros(11) for _ in test], columns=CANONICAL_GRAIN_LABELS)
    sample.insert(0, "sample_id", test)
    rows = []
    for sid in train + test:
        for camera in ("z", "a"):
            path = tmp_path / f"{sid}_{camera}.png"
            Image.new("RGB", (100, 100), (3, 4, 5)).save(path)
            rows.append({"sample_id": sid, "split": "train" if sid in train else "test",
                         "camera": camera, "path": str(path)})
    cameras = pd.DataFrame([{"phone": c, "width": 100, "height": 100, "ppm": 1}
                            for c in ("a", "z")]).set_index("phone")
    return truth, sample, pd.DataFrame(rows).sample(frac=1, random_state=4), cameras


def fixture(tmp_path, recipe, request, progress=lambda *args, **kwargs: None):
    from soilgrain.evidence_visual_language import _run

    truth, sample, photos, cameras = inputs(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("fixed")
    item = {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
    identity = {"client_version": "codex-cli 0.154.0", "producing_commit": "fixture",
                "code_sources": [item]}
    output = tmp_path / "experiments" / "september23" / recipe
    output.mkdir(parents=True, exist_ok=True)
    return _run(output, recipe, truth, sample, photos, cameras, [item], identity,
                request, progress), output


class Request:
    def __init__(self, *, usage=100, mode="ok"):
        self.calls = 0
        self.usage = usage
        self.mode = mode

    def __call__(self, command, *, input, text, stdout, stderr, timeout):
        self.calls += 1
        folder = Path(command[command.index("--cd") + 1])
        output = folder.parents[1]
        receipt = output / "receipts" / f"{folder.name}.json"
        assert receipt.exists() and json.loads((output / "requests.json").read_text())["requests"][-1]["state"] == "requested"
        assert timeout == 300 and '-c' in command and 'model="gpt-6-astra"' in command
        assert "S00" not in input and "T00" not in input
        response = json.dumps({"cdf": list(range(0, 101, 10))})
        (folder / "response.json").write_text(response)
        events = [{"type": "item.completed", "item": {"type": "agent_message", "text": response}},
                  {"type": "turn.completed", "usage": {"input_tokens": self.usage,
                   "output_tokens": 20, "reasoning_output_tokens": 5}}]
        if self.mode == "transport":
            events.insert(0, {"type": "error", "message": "Reconnecting... 2/5 (stream disconnected before completion: websocket closed by server before response.completed)"})
            stderr.write("2026-09-23T00:00:00Z  WARN codex_core::responses_retry: stream disconnected - retrying sampling request (1/5 in 209ms)... turn_id=01a0c9b9-7874-7183-a636-570385e59d4d retries=1 max_retries=5 sampling_error=stream disconnected before completion: WebSocket protocol error: Connection reset without closing handshake\n")
        stdout.write("\n".join(json.dumps(e) for e in events))
        return SimpleNamespace(returncode=0)


def test_paired_examples_group_one_label_per_soil_and_exclude_holdout(tmp_path):
    from soilgrain.evidence_visual_language import request_context
    from soilgrain.visual_language import build_prompt

    truth, _, photos, _ = inputs(tmp_path)
    prompt, selection, selected = request_context("paired_examples_vlm", truth, photos, "S03")
    _, examples = build_prompt(truth, "S03", 2)
    assert selection["examples"] == examples and len(selected) == 18
    assert [(row.sample_id, row.camera) for row, _ in selected[:4]] == [
        (examples[0], "a"), (examples[0], "z"), (examples[1], "a"), (examples[1], "z")]
    assert [(row.sample_id, row.camera) for row, _ in selected[-2:]] == [("S03", "a"), ("S03", "z")]
    assert prompt.count("mass percentages passing:") == 8
    assert "images 1-2" in prompt and "images 15-16" in prompt
    assert "S03" not in prompt and "S00" not in prompt
    truth.loc[truth.sample_id == "S03", CANONICAL_GRAIN_LABELS] = 100
    changed, changed_selection, _ = request_context("paired_examples_vlm", truth, photos, "S03")
    assert changed == prompt and changed_selection == selection


def test_complete_context_reuses_scientific_prompt_and_image_recipe(tmp_path):
    from soilgrain.context_visual_language import request_context as old_context
    from soilgrain.evidence_visual_language import request_context

    truth, _, photos, _ = inputs(tmp_path)
    prompt, selection, rows = request_context("complete_context_vlm", truth, photos, "S03")
    old_prompt, old_selection, old_rows = old_context("all_examples_vlm", truth, photos, "S03")
    assert (prompt, selection) == (old_prompt, old_selection)
    assert [(row.path, width) for row, width in rows] == [
        (row.path, width) for row, width in old_rows]


def test_paired_examples_require_each_selected_soil_to_have_a_view(tmp_path):
    from soilgrain.evidence_visual_language import request_context
    from soilgrain.visual_language import build_prompt

    truth, _, photos, _ = inputs(tmp_path)
    _, examples = build_prompt(truth, "S03", 2)
    photos = photos.loc[photos.sample_id != examples[0]]
    with pytest.raises(ValueError, match="Every example"):
        request_context("paired_examples_vlm", truth, photos, "S03")


def test_transport_validator_accepts_only_recognized_recovery_with_single_final(tmp_path):
    from soilgrain.evidence_visual_language import response_usage

    folder = tmp_path
    response = '{"cdf": [0,10,20,30,40,50,60,70,80,90,100]}'
    (folder / "response.json").write_text(response)
    (folder / "stderr.txt").write_text("")
    reconnect = {"type": "error", "message": "Reconnecting... 2/5 (stream disconnected before completion: websocket closed by server before response.completed)"}
    fallback = {"type": "item.completed", "item": {"type": "error", "message": "Falling back from WebSockets to HTTPS transport. stream disconnected before completion: websocket closed by server before response.completed"}}
    code = {"type": "item.completed", "item": {"type": "error", "message": "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."}}
    final = {"type": "item.completed", "item": {"type": "agent_message", "text": response}}
    done = {"type": "turn.completed", "usage": {"input_tokens": 9, "output_tokens": 4, "reasoning_output_tokens": 1}}
    def write(events):
        (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    write([code, reconnect, fallback, final, done])
    assert response_usage(folder)["transport_events"] == 2
    for bad in ({"type": "error", "message": "authentication failed"},
                {"type": "item.completed", "item": {"type": "error", "message": "Code Mode is unavailable: unrelated failure"}},
                {"type": "item.completed", "item": {"type": "command_execution"}},
                {"type": "turn.failed"}, final, done):
        write([reconnect, fallback, final, done, bad])
        with pytest.raises(ValueError):
            response_usage(folder)


def test_stderr_transport_audit_accepts_known_retry_and_rejects_unknown(tmp_path):
    from soilgrain.evidence_visual_language import stderr_transport_usage

    stderr = tmp_path / "stderr.txt"
    prefix = "2026-09-23T00:00:00Z  WARN "
    known = "codex_core::responses_retry: stream disconnected - retrying sampling request (1/5 in 209ms)... turn_id=01a0c9b9-7874-7183-a636-570385e59d4d retries=1 max_retries=5 sampling_error=stream disconnected before completion: WebSocket protocol error: Connection reset without closing handshake"
    stderr.write_text(prefix + known + "\n" + prefix + "codex_core::client: falling back to HTTP\n")
    assert stderr_transport_usage(stderr) == {"stderr_retry_warnings": 1,
                                            "stderr_fallback_warnings": 1}
    stderr.write_text(prefix + known.replace("Connection reset without closing handshake", "authentication failed"))
    with pytest.raises(ValueError):
        stderr_transport_usage(stderr)
    stderr.write_text("2026-09-23T00:00:00Z  ERROR codex_core::client: unauthorized")
    with pytest.raises(ValueError):
        stderr_transport_usage(stderr)


@pytest.mark.parametrize("recipe", ["paired_examples_vlm", "complete_context_vlm"])
def test_complete_run_exports_34_rows_and_records_transport(tmp_path, recipe):
    request = Request(mode="transport")
    summary, output = fixture(tmp_path, recipe, request)
    assert summary["eligible"] is True and request.calls == summary["requests"] == 34
    assert summary["transport_events"] == 34 and summary["input_tokens"] == 3400
    assert summary["stderr_retry_warnings"] == 34
    assert len(pd.read_csv(summary["oof"]["path"])) == 24
    assert len(pd.read_csv(summary["submission"]["path"])) == 10
    assert not (output / "failure.json").exists()


def test_budget_stops_before_second_dispatch_and_never_reissues(tmp_path):
    request = Request(usage=2_000_000)
    with pytest.raises(ValueError, match="budget"):
        fixture(tmp_path, "paired_examples_vlm", request)
    output = tmp_path / "experiments" / "september23" / "paired_examples_vlm"
    assert request.calls == 1 and json.loads((output / "failure.json").read_text())["eligible"] is False
    with pytest.raises(ValueError, match="retry"):
        fixture(tmp_path, "paired_examples_vlm", request)
    assert request.calls == 1


def test_resume_rejects_tampered_record_before_dispatch(tmp_path):
    from soilgrain.evidence_visual_language import _run

    request = Request()
    _, output = fixture(tmp_path, "complete_context_vlm", request)
    ledger = json.loads((output / "requests.json").read_text())
    ledger["requests"][0]["examples"].reverse()
    (output / "requests.json").write_text(json.dumps(ledger))
    truth, sample, photos, cameras = inputs(tmp_path)
    source = tmp_path / "source.txt"
    item = {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
    identity = {"client_version": "codex-cli 0.154.0", "producing_commit": "fixture",
                "code_sources": [item]}
    with pytest.raises(ValueError):
        _run(output, "complete_context_vlm", truth, sample, photos, cameras, [item], identity, request)
    assert request.calls == 34


def test_failed_final_still_records_completed_usage_and_never_retries(tmp_path):
    class BadFinal(Request):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            folder = Path(command[command.index("--cd") + 1])
            (folder / "response.json").write_text('{"cdf": [1, 0]}')
            return result

    request = BadFinal(mode="transport")
    with pytest.raises(ValueError):
        fixture(tmp_path, "paired_examples_vlm", request)
    output = tmp_path / "experiments" / "september23" / "paired_examples_vlm"
    record = json.loads((output / "requests.json").read_text())["requests"][0]
    failure = json.loads((output / "failure.json").read_text())
    assert record["state"] == "failed" and record["input_tokens"] == 100
    assert record["transport_events"] == failure["transport_events"] == 1
    assert failure["input_tokens"] == 100
    with pytest.raises(ValueError, match="retry"):
        fixture(tmp_path, "paired_examples_vlm", request)
    assert request.calls == 1


def test_verified_completed_prefix_resumes_without_repeating_first_dispatch(tmp_path):
    request = Request()

    def stop_after_one(*args, **kwargs):
        raise KeyboardInterrupt("between requests")

    with pytest.raises(KeyboardInterrupt):
        fixture(tmp_path, "paired_examples_vlm", request, stop_after_one)
    output = tmp_path / "experiments" / "september23" / "paired_examples_vlm"
    assert request.calls == 1 and not (output / "failure.json").exists()
    summary, _ = fixture(tmp_path, "paired_examples_vlm", request)
    assert summary["eligible"] is True and request.calls == 34


def test_exception_after_completed_turn_still_records_usage(tmp_path):
    class Interrupted(Request):
        def __call__(self, command, **kwargs):
            super().__call__(command, **kwargs)
            raise TimeoutError("client timed out after final event")

    request = Interrupted()
    with pytest.raises(TimeoutError):
        fixture(tmp_path, "paired_examples_vlm", request)
    output = tmp_path / "experiments" / "september23" / "paired_examples_vlm"
    record = json.loads((output / "requests.json").read_text())["requests"][0]
    assert record["state"] == "failed" and record["input_tokens"] == 100
