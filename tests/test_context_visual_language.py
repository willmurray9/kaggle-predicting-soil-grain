import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.constants import CANONICAL_GRAIN_LABELS


def label_frame(ids):
    frame = pd.DataFrame([np.linspace(i, 100, 11) for i in range(len(ids))],
                         columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, "sample_id", ids)
    return frame


def test_twenty_five_mm_crop_comes_directly_from_original_pixels(tmp_path):
    from soilgrain.context_visual_language import prepare_physical_crop
    from soilgrain.visual_language import prepare_crop

    values = np.arange(2000, dtype=np.uint16)[None, :] % 256
    pixels = np.repeat(values, 2000, axis=0).astype(np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(np.dstack([pixels, pixels, pixels]), "RGB").save(source)
    camera = pd.Series({"width": 2000, "height": 2000, "ppm": 10})
    broad, original_broad = tmp_path / "broad.png", tmp_path / "original-broad.png"
    close = tmp_path / "close.png"

    prepare_physical_crop(source, camera, broad, 100)
    prepare_crop(source, camera, original_broad)
    prepare_physical_crop(source, camera, close, 25)

    assert broad.read_bytes() == original_broad.read_bytes()
    with Image.open(broad) as image:
        assert image.size == (768, 768)
    with Image.open(close) as image:
        assert image.size == (250, 250)
        np.testing.assert_array_equal(np.asarray(image)[:, :, 0], pixels[875:1125, 875:1125])


def context_inputs(tmp_path):
    train_ids = [f"S{i:02d}" for i in range(24)]
    test_ids = [f"T{i:02d}" for i in range(10)]
    truth = label_frame(train_ids[::-1])
    sample = label_frame(test_ids)
    rows = []
    for number, sample_id in enumerate(train_ids + test_ids):
        for camera, suffix in [("z_camera", "b"), ("a_camera", "a")]:
            path = tmp_path / f"{sample_id}_{suffix}.png"
            Image.new("RGB", (100, 100), (number, number, number)).save(path)
            rows.append({"sample_id": sample_id,
                         "split": "train" if sample_id.startswith("S") else "test",
                         "camera": camera, "path": str(path)})
    photos = pd.DataFrame(rows).sample(frac=1, random_state=7)
    cameras = pd.DataFrame([
        {"phone": "a_camera", "width": 100, "height": 100, "ppm": 1},
        {"phone": "z_camera", "width": 100, "height": 100, "ppm": 1},
    ]).set_index("phone")
    return truth, sample, photos, cameras


def test_all_examples_excludes_holdout_sorts_examples_and_hides_query_label(tmp_path):
    from soilgrain.context_visual_language import request_context

    truth, _, photos, _ = context_inputs(tmp_path)
    prompt, selection, selected = request_context("all_examples_vlm", truth, photos, "S03")
    expected = [f"S{i:02d}" for i in range(24) if i != 3]
    assert selection["examples"] == expected
    assert [row.sample_id for row, _ in selected[:23]] == expected
    assert all(Path(row.path).name.endswith("_a.png") for row, _ in selected[:23])
    assert [(row.camera, Path(row.path).name) for row, _ in selected[23:]] == [
        ("a_camera", "S03_a.png"), ("z_camera", "S03_b.png")]
    assert all(width == 100 for _, width in selected)
    assert "S03" not in prompt and "S00" not in prompt

    truth.loc[truth.sample_id == "S03", CANONICAL_GRAIN_LABELS] = 100
    changed, changed_selection, _ = request_context("all_examples_vlm", truth, photos, "S03")
    assert changed == prompt
    assert changed_selection == selection


def test_all_examples_uses_all_twenty_four_training_soils_for_test_query(tmp_path):
    from soilgrain.context_visual_language import request_context

    truth, _, photos, _ = context_inputs(tmp_path)
    prompt, selection, selected = request_context("all_examples_vlm", truth, photos, "T00")
    assert selection["examples"] == [f"S{i:02d}" for i in range(24)]
    assert len(selected) == 26
    assert "first 24 images" in prompt


def test_multiscale_pairs_broad_then_close_crops_for_every_source_photo(tmp_path):
    from soilgrain.context_visual_language import request_context
    from soilgrain.visual_language import build_prompt

    truth, _, photos, _ = context_inputs(tmp_path)
    prompt, selection, selected = request_context("multiscale_vlm", truth, photos, "S03")
    _, expected_examples = build_prompt(truth, "S03", 2)
    assert selection["examples"] == expected_examples
    assert len(selected) == 20
    for first, second in zip(selected[::2], selected[1::2], strict=True):
        assert first[0].path == second[0].path
        assert (first[1], second[1]) == (100, 25)
    assert "consecutive image pair" in prompt
    assert "100 mm then 25 mm" in prompt
    assert "query photo 1" in prompt.lower() and "query photo 2" in prompt.lower()
    truth.loc[truth.sample_id == "S03", CANONICAL_GRAIN_LABELS] = 100
    changed, changed_selection, _ = request_context("multiscale_vlm", truth, photos, "S03")
    assert changed == prompt and changed_selection == selection


class FakeRequest:
    def __init__(self, mode="ok", input_tokens=100, output_tokens=20):
        self.mode = mode
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def __call__(self, command, *, input, text, stdout, stderr, timeout):
        self.calls += 1
        assert "--ignore-rules" in command and "--ephemeral" in command
        folder = Path(command[command.index("--cd") + 1])
        assert folder.name.startswith("query_")
        assert all(not path.name.startswith(("S", "T")) for path in folder.iterdir())
        assert not any(value in input for value in ["S00", "S23", "T00"])
        output = folder.parents[1]
        receipt = output / "receipts" / f"{folder.name}.json"
        ledger = json.loads((output / "requests.json").read_text())
        assert receipt.is_file() and ledger["requests"][-1]["state"] == "requested"
        response = json.dumps({"cdf": list(range(0, 101, 10))})
        if self.mode == "malformed":
            response = '{"cdf": [1, 0]}'
        (folder / "response.json").write_text(response)
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": response}},
            {"type": "turn.completed", "usage": {
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "reasoning_output_tokens": 5}},
        ]
        stdout.write("\n".join(json.dumps(event) for event in events))
        return SimpleNamespace(returncode=0)


def run_fixture(tmp_path, recipe, request):
    from soilgrain.context_visual_language import _run

    truth, sample, photos, cameras = context_inputs(tmp_path)
    source = tmp_path / "verified-source.txt"
    source.write_text("fixed")
    source_record = {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
    identity = {"client_version": "codex-cli 0.154.0", "producing_commit": "fixture",
                "code_sources": [source_record]}
    output = tmp_path / "experiments" / "september22" / recipe
    output.mkdir(parents=True)
    return (_run(output, recipe, truth, sample, photos, cameras, [source_record], identity,
                 request), output, truth, sample)


@pytest.mark.parametrize("recipe", ["multiscale_vlm", "all_examples_vlm"])
def test_complete_recipe_requires_thirty_four_curves_and_writes_valid_template(tmp_path, recipe):
    request = FakeRequest()
    summary, output, truth, sample = run_fixture(tmp_path, recipe, request)
    assert summary["experiment"] == recipe and summary["eligible"] is True
    assert summary["heldout_soils"] == 24 and summary["test_soils"] == 10
    assert summary["requests"] == request.calls == 34
    assert summary["input_tokens"] == 3400 and summary["output_tokens"] == 680
    oof = pd.read_csv(summary["oof"]["path"])
    submission = pd.read_csv(summary["submission"]["path"])
    assert oof.sample_id.tolist() == truth.sample_id.tolist()
    assert submission.sample_id.tolist() == sample.sample_id.tolist()
    assert submission.columns.tolist() == sample.columns.tolist()
    assert np.array_equal(submission[list(CANONICAL_GRAIN_LABELS)].to_numpy(),
                          np.tile(np.arange(0, 101, 10), (10, 1)))
    assert not (output / "failure.json").exists()


def test_failed_request_is_recorded_and_never_retried(tmp_path):
    from soilgrain.context_visual_language import _run

    request = FakeRequest(mode="malformed")
    with pytest.raises(ValueError):
        run_fixture(tmp_path, "multiscale_vlm", request)
    output = tmp_path / "experiments" / "september22" / "multiscale_vlm"
    ledger = json.loads((output / "requests.json").read_text())
    assert request.calls == 1 and ledger["requests"][0]["state"] == "failed"
    assert json.loads((output / "failure.json").read_text())["eligible"] is False

    truth, sample, photos, cameras = context_inputs(tmp_path)
    source = tmp_path / "verified-source.txt"
    source_record = {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
    identity = {"client_version": "codex-cli 0.154.0", "producing_commit": "fixture",
                "code_sources": [source_record]}
    with pytest.raises(ValueError, match="retry"):
        _run(output, "multiscale_vlm", truth, sample, photos, cameras,
             [source_record], identity, request)
    assert request.calls == 1


def test_aggregate_budget_stops_before_a_second_request(tmp_path):
    request = FakeRequest(input_tokens=2_000_000)
    with pytest.raises(ValueError, match="budget"):
        run_fixture(tmp_path, "all_examples_vlm", request)
    output = tmp_path / "experiments" / "september22" / "all_examples_vlm"
    failure = json.loads((output / "failure.json").read_text())
    assert request.calls == 1 and failure["eligible"] is False


def test_partial_preparation_fails_closed_without_dispatch_or_retry(tmp_path):
    from soilgrain.context_visual_language import _run

    truth, sample, photos, cameras = context_inputs(tmp_path)
    source = tmp_path / "verified-source.txt"
    source.write_text("fixed")
    source_record = {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
    identity = {"client_version": "codex-cli 0.154.0", "producing_commit": "fixture",
                "code_sources": [source_record]}
    output = tmp_path / "experiments" / "september22" / "multiscale_vlm"
    output.mkdir(parents=True)
    request = FakeRequest()
    cameras["ppm"] = 2

    with pytest.raises(ValueError, match="does not fit"):
        _run(output, "multiscale_vlm", truth, sample, photos, cameras,
             [source_record], identity, request)
    assert request.calls == 0 and (output / "failure.json").is_file()
    cameras["ppm"] = 1
    with pytest.raises(ValueError, match="retry"):
        _run(output, "multiscale_vlm", truth, sample, photos, cameras,
             [source_record], identity, request)
    assert request.calls == 0


def test_resume_rejects_context_mismatch_without_new_request(tmp_path):
    from soilgrain.context_visual_language import _run

    request = FakeRequest()
    summary, output, truth, sample = run_fixture(tmp_path, "all_examples_vlm", request)
    assert summary["eligible"]
    ledger_path = output / "requests.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["requests"][0]["examples"].reverse()
    ledger_path.write_text(json.dumps(ledger))
    source = tmp_path / "verified-source.txt"
    source_record = {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
    identity = {"client_version": "codex-cli 0.154.0", "producing_commit": "fixture",
                "code_sources": [source_record]}
    with pytest.raises(ValueError):
        _run(output, "all_examples_vlm", truth, sample, context_inputs(tmp_path)[2],
             context_inputs(tmp_path)[3], [source_record], identity, request)
    assert request.calls == 34
    assert not (output / "failure.json").exists()


def test_completed_prefix_resumes_after_between_request_interruption(tmp_path):
    from soilgrain.context_visual_language import _run

    truth, sample, photos, cameras = context_inputs(tmp_path)
    source = tmp_path / "verified-source.txt"
    source.write_text("fixed")
    source_record = {"path": str(source), "sha256": sha256(source.read_bytes()).hexdigest()}
    identity = {"client_version": "codex-cli 0.154.0", "producing_commit": "fixture",
                "code_sources": [source_record]}
    output = tmp_path / "experiments" / "september22" / "multiscale_vlm"
    output.mkdir(parents=True)
    request = FakeRequest()

    def interrupt_between_requests(*args, **kwargs):
        raise KeyboardInterrupt("operator stopped after completion")

    with pytest.raises(KeyboardInterrupt):
        _run(output, "multiscale_vlm", truth, sample, photos, cameras,
             [source_record], identity, request, interrupt_between_requests)
    ledger = json.loads((output / "requests.json").read_text())
    assert len(ledger["requests"]) == 1 and ledger["requests"][0]["state"] == "complete"
    assert not (output / "failure.json").exists()

    summary = _run(output, "multiscale_vlm", truth, sample, photos, cameras,
                   [source_record], identity, request, lambda *args, **kwargs: None)
    assert summary["eligible"] is True and request.calls == 34
    assert not (output / "failure.json").exists()
