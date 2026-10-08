import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain import october2_visual_language as october2
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.visual_language import cli_command, prepare_crop as original_crop


CAMERAS = pd.DataFrame({"phone": ["A", "B"], "camera": ["A", "B"], "width": [800, 800],
                        "height": [600, 600], "ppm": [4.6, 5.5]}).set_index("phone")


def soil_data(tmp_path, train=9):
    ids = [f"S{i}" for i in range(train)]
    rows, photos = [], []
    for i, sample_id in enumerate(ids):
        rows.append([sample_id, *np.linspace(i, 100, 11)])
        for camera in ("A", "B"):
            path = tmp_path / f"{sample_id}_{camera}.png"
            Image.new("RGB", (800, 600), (i * 20, 50, 90)).save(path)
            photos.append({"split": "train", "sample_id": sample_id, "camera": camera, "path": str(path)})
    for view in range(3):
        path = tmp_path / f"T_{view}.png"
        Image.new("RGB", (800, 600), (5, 5 * view, 5)).save(path)
        photos.append({"split": "test", "sample_id": "T", "camera": "B", "path": str(path)})
    truth = pd.DataFrame(rows, columns=["sample_id", *CANONICAL_GRAIN_LABELS])
    sample = pd.DataFrame([["T", *[0.0] * 11]], columns=["sample_id", *CANONICAL_GRAIN_LABELS])
    return truth, sample, pd.DataFrame(photos)


def test_low_reasoning_codex_command_is_the_original_command():
    folder, images, schema = Path("/q"), [Path("/q/image_01.png")], Path("/s.json")
    assert october2.codex_command(folder, images, schema, "low") == cli_command(folder, images, schema)
    high = october2.codex_command(folder, images, schema, "high")
    assert 'model_reasoning_effort="high"' in high and 'model_reasoning_effort="low"' not in high


def test_crop_matches_original_at_768_and_caps_matched_resolution(tmp_path):
    source = tmp_path / "source.png"
    Image.effect_noise((1000, 900), 40).convert("RGB").save(source)
    camera = pd.Series({"ppm": 8.0, "width": 1000, "height": 900})
    original_crop(source, camera, tmp_path / "original.png")
    october2.prepare_crop(source, camera, tmp_path / "new.png", 768)
    october2.prepare_crop(source, camera, tmp_path / "matched.png", 460)
    assert (tmp_path / "original.png").read_bytes() == (tmp_path / "new.png").read_bytes()
    assert Image.open(tmp_path / "new.png").size == (768, 768)
    assert Image.open(tmp_path / "matched.png").size == (460, 460)


def test_views_follow_recipe_and_exclude_the_query(tmp_path):
    truth, _sample, photos = soil_data(tmp_path)
    for recipe, expected in (("rerun_a_vlm", 1), ("all_views_vlm", 3)):
        prompt, selection, rows = october2.request_context(recipe, truth, photos, "T")
        assert len(rows) == 8 + expected and f"remaining {expected} query" in prompt
        assert all(row.sample_id == "T" for row in rows[8:])
    _prompt, selection, rows = october2.request_context("all_views_vlm", truth, photos, "S0")
    assert "S0" not in selection["examples"] and len(rows) == 8 + 2


def stream(curve, model=october2.CLAUDE_MODEL, tool="StructuredOutput"):
    return [
        {"type": "system", "subtype": "init", "model": model, "tools": ["StructuredOutput"],
         "mcp_servers": []},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": tool,
                                                       "input": {"cdf": curve}}]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "structured_output": {"cdf": curve}, "modelUsage": {model: {}}, "total_cost_usd": 0.1,
         "usage": {"input_tokens": 3, "cache_creation_input_tokens": 100,
                   "cache_read_input_tokens": 0, "output_tokens": 40,
                   "output_tokens_details": {"thinking_tokens": 5}}},
    ]


def fake_claude(curves, cost=0.1):
    calls = []

    def run(command, input, text, stdout, stderr, timeout, cwd, start_new_session):
        assert command[0] == october2.CLAUDE and "--tools" in command and cwd.is_dir()
        assert start_new_session
        message = json.loads(input)["message"]["content"]
        assert [block["type"] for block in message][-1] == "text"
        curve = curves[len(calls) % len(curves)]
        calls.append(cwd)
        if curve is None:
            return type("Result", (), {"returncode": 1})()
        if curve == "launch":
            raise FileNotFoundError("missing client")
        events = stream(curve)
        events[-1]["total_cost_usd"] = cost
        stdout.write("\n".join(json.dumps(event) for event in events) + "\n")
        return type("Result", (), {"returncode": 0})()

    return run, calls


def test_claude_stream_validation_rejects_tools_models_and_mismatches(tmp_path):
    curve = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    for events, ok in ((stream(curve), True), (stream(curve, model="other"), False),
                       (stream(curve, tool="Read"), False)):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        (folder / "response.json").write_text(json.dumps({"cdf": curve}))
        if ok:
            assert october2.claude_response_usage(folder) == {
                "input_tokens": 103, "output_tokens": 40, "reasoning_output_tokens": 5,
                "cost_usd": 0.1}
        else:
            with pytest.raises(ValueError):
                october2.claude_response_usage(folder)
    (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in stream(curve)))
    (folder / "response.json").write_text(json.dumps({"cdf": [0] * 10 + [100]}))
    with pytest.raises(ValueError):
        october2.claude_response_usage(folder)


def test_run_retries_objective_failures_once_then_finalizes(tmp_path):
    truth, sample, photos = soil_data(tmp_path)
    good = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    run, calls = fake_claude([None, good] + [good] * 20)
    output = tmp_path / "out"
    output.mkdir()
    ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    summary = october2._run(output, "claude_vlm", truth, sample, photos, CAMERAS, ids,
                            run=run, progress=lambda *a, **k: None)
    assert summary["eligible"] and summary["attempts"] == len(ids) + 1
    assert summary["failed_attempts"] == 1
    assert len(list((output / "requests").glob("query_*/attempt_2/response.json"))) == 1
    assert pd.read_csv(output / "claude_vlm.csv").sample_id.tolist() == ["T"]


def test_run_marks_recipe_ineligible_after_two_failures(tmp_path):
    truth, sample, photos = soil_data(tmp_path)
    run, calls = fake_claude([None])
    output = tmp_path / "out"
    output.mkdir()
    ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    with pytest.raises(ValueError, match="ineligible"):
        october2._run(output, "claude_vlm", truth, sample, photos, CAMERAS, ids,
                      run=run, progress=lambda *a, **k: None)
    assert len(calls) < 2 * len(ids)  # Stops spending once a query exhausts both attempts.
    failure = json.loads((output / "failure.json").read_text())
    assert failure["eligible"] is False and failure["cost_usd"] == pytest.approx(2.0 * len(calls))


def test_rejected_responses_are_charged_and_stop_at_the_budget(tmp_path):
    truth, sample, photos = soil_data(tmp_path)
    rejected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 100]  # Non-monotone: an objective failure.
    run, calls = fake_claude([rejected], cost=1.5)
    output = tmp_path / "out"
    output.mkdir()
    ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    original = october2.BUDGETS["claude"]["usd"]
    october2.BUDGETS["claude"]["usd"] = 3.0
    try:
        with pytest.raises(ValueError):
            october2._run(output, "claude_vlm", truth, sample, photos, CAMERAS, ids,
                          run=run, progress=lambda *a, **k: None)
    finally:
        october2.BUDGETS["claude"]["usd"] = original
    records = [json.loads(p.read_text()) for p in (output / "records").glob("*.json")]
    assert len(calls) <= 2 + october2.BUDGETS["claude"]["workers"]
    assert all(r["state"] == "failed" and r["cost_usd"] == 1.5 for r in records)


def test_launch_errors_do_not_consume_attempts_and_resume_uses_dispatched_responses(tmp_path):
    truth, sample, photos = soil_data(tmp_path)
    good = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    output = tmp_path / "out"
    output.mkdir()
    ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    run, _ = fake_claude(["launch"])
    with pytest.raises(october2.LaunchError):
        october2._run(output, "claude_vlm", truth, sample, photos, CAMERAS, ids,
                      run=run, progress=lambda *a, **k: None)
    assert not list((output / "records").glob("*.json"))
    assert not list((output / "requests").glob("*/*/events.jsonl"))
    # Simulate a crash after a dispatch completed but before its record was saved.
    folder = output / "requests" / "query_01" / "attempt_1"
    run, calls = fake_claude([good])
    october2.dispatch("claude_vlm", folder, sorted(folder.glob("image_*.png")),
                      (folder / "prompt.txt").read_text(), output / "schema.json", run) \
        if (folder / "prompt.txt").exists() else None
    run, calls = fake_claude([good])
    summary = october2._run(output, "claude_vlm", truth, sample, photos, CAMERAS, ids,
                            run=run, progress=lambda *a, **k: None)
    assert summary["eligible"] and summary["failed_attempts"] == 0
    assert len(calls) + (1 if (folder / "events.jsonl").exists() else 0) == len(ids)


def test_wide_crops_scale_the_physical_width_and_only_the_width_sentence(tmp_path):
    source = tmp_path / "source.png"
    Image.effect_noise((1000, 900), 40).convert("RGB").save(source)
    camera = pd.Series({"ppm": 4.6, "width": 1000, "height": 900})
    october2.prepare_crop(source, camera, tmp_path / "wide.png", 768, 150)
    assert Image.open(tmp_path / "wide.png").size == (690, 690)
    truth, _sample, photos = soil_data(tmp_path)
    narrow, *_ = october2.request_context("all_views_b_vlm", truth, photos, "T")
    wide, *_ = october2.request_context("wide_all_views_a_vlm", truth, photos, "T")
    assert wide == narrow.replace("is a 100 mm square", "is a 150 mm square") and wide != narrow


def test_recipe_mean_requires_finished_components_and_a_minimum(tmp_path, monkeypatch):
    truth, sample, _photos = soil_data(tmp_path)
    cfg = type("Cfg", (), {"artifacts_dir": tmp_path / "artifacts"})()
    monkeypatch.setattr(october2, "load_inputs", lambda _config: (cfg, truth, sample, None, None, []))
    base = tmp_path / "artifacts" / "experiments" / "october2"
    ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    for recipe, level in (("wide_all_views_a_vlm", 10), ("wide_all_views_b_vlm", 30)):
        curves = np.array([[level] * 10 + [100]] * len(ids), dtype=float)
        october2.finalize(base / recipe, recipe, truth, sample, curves, {})
    with pytest.raises(ValueError, match="exactly one"):
        october2.recipe_mean("wide_draw_mean")  # wide_c has neither a summary nor a failure.
    (base / "wide_all_views_c_vlm").mkdir(parents=True)
    (base / "wide_all_views_c_vlm" / "failure.json").write_text(json.dumps({"eligible": False}))
    summary = october2.recipe_mean("wide_draw_mean")
    assert summary["draws"] == 2 and summary["excluded"] == ["wide_all_views_c_vlm"]
    result = pd.read_csv(base / "wide_draw_mean" / "wide_draw_mean.csv")
    np.testing.assert_allclose(result.iloc[0, 1:11].to_numpy(dtype=float), [20] * 10)
    (base / "claude_all_views_a_vlm").mkdir(parents=True)
    for d in "abc":
        folder = base / f"claude_all_views_{d}_vlm"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "failure.json").write_text(json.dumps({"eligible": False}))
    with pytest.raises(ValueError, match="at least 2"):
        october2.recipe_mean("claude_all_views_mean")


def test_wide_recipes_omit_only_the_declared_non_soil_photo(tmp_path, monkeypatch):
    truth, _sample, photos = soil_data(tmp_path)
    excluded = october2.file_record(Path(photos.loc[photos.sample_id == "T"].path.iloc[1]))["sha256"]
    monkeypatch.setitem(october2.RECIPES["wide_all_views_a_vlm"], "exclude_sha256", (excluded,))
    prompt, selection, rows = october2.request_context("wide_all_views_a_vlm", truth, photos, "T")
    assert "remaining 2 query" in prompt and len(rows) == 8 + 2
    assert excluded not in [source["sha256"] for source in selection["image_sources"]]
    _prompt, _selection, rows = october2.request_context("all_views_b_vlm", truth, photos, "T")
    assert len(rows) == 8 + 3


def test_seeded_panels_change_only_example_lines_and_exclude_the_query(tmp_path):
    from soilgrain.visual_language import build_prompt
    truth, _sample, _photos = soil_data(tmp_path)
    assert october2.seeded_prompt(truth, "S0", 2, 0) == build_prompt(truth, "S0", 2)
    original, _ = build_prompt(truth, "S0", 2)
    prompt, examples = october2.seeded_prompt(truth, "S0", 2, 3)
    assert "S0" not in examples and len(set(examples)) == 8
    changed = [i for i, (a, b) in enumerate(zip(original.splitlines(), prompt.splitlines())) if a != b]
    assert changed and set(changed) <= set(range(1, 9))


def test_tiles_shift_along_the_long_side_and_count_in_the_prompt(tmp_path):
    source = tmp_path / "source.png"
    Image.effect_noise((1000, 500), 40).convert("RGB").save(source)
    camera = pd.Series({"ppm": 4.0, "width": 1000, "height": 500})
    october2.prepare_crop(source, camera, tmp_path / "center.png", 768)
    october2.prepare_crop(source, camera, tmp_path / "left.png", 768, 100, -50)
    assert (np.asarray(Image.open(tmp_path / "left.png"))
            == np.asarray(Image.open(source).crop((100, 50, 500, 450)))).all()
    with pytest.raises(ValueError, match="Tile does not fit"):
        october2.prepare_crop(source, camera, tmp_path / "far.png", 768, 100, -80)
    truth, _sample, photos = soil_data(tmp_path)
    prompt, selection, rows = october2.request_context("tiled_all_views_a_vlm", truth, photos, "T")
    assert len(rows) == 8 + 6 and "remaining 6 query" in prompt
    assert "100 mm square crop." in prompt and "center crop" not in prompt
    assert [s.get("offset_mm") for s in selection["image_sources"][8:]] == [-50, 50] * 3


def test_exposure_gain_scales_after_resize_and_is_absent_by_default(tmp_path):
    source = tmp_path / "source.png"
    Image.effect_noise((1000, 900), 40).convert("RGB").save(source)
    camera = pd.Series({"ppm": 8.0, "width": 1000, "height": 900})
    october2.prepare_crop(source, camera, tmp_path / "plain.png", 460)
    october2.prepare_crop(source, camera, tmp_path / "none.png", 460, 100, 0, None)
    october2.prepare_crop(source, camera, tmp_path / "gain.png", 460, 100, 0, 0.8)
    assert (tmp_path / "plain.png").read_bytes() == (tmp_path / "none.png").read_bytes()
    plain = np.asarray(Image.open(tmp_path / "plain.png"), dtype=float)
    gained = np.asarray(Image.open(tmp_path / "gain.png"), dtype=float)
    np.testing.assert_array_equal(gained, np.clip(np.rint(plain * 0.8), 0, 255))


def test_october7_recipes_keep_the_prompt_and_route_outputs():
    for recipe in ("matched_all_views_a_vlm", "exposure_all_views_a_vlm", "aligned_all_views_b_vlm",
                   "all_views_d_vlm"):
        spec = october2.RECIPES[recipe]
        assert spec["views"] == "all" and spec.get("seed", 0) == 0 and october2.round_dir(recipe) == "october7r"
    assert october2.round_dir("all_views_five_draw_mean") == "october7r"
    assert october2.round_dir("all_views_vlm") == "october2"
    assert set(october2.EXPOSURE_GAINS) == {"Motorola Edge", "Motorola Edge 60 Fusion", "Samsung A52",
                                            "iPhone 14", "iPhone 16"}


def codex_stream(tmp_path, extra_errors):
    folder = tmp_path / "attempt"
    folder.mkdir(parents=True)
    response = json.dumps({"cdf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]})
    events = [{"type": "thread.started"}]
    events += [{"type": "item.completed", "item": {"type": "error", "message": m}} for m in extra_errors]
    events += [{"type": "turn.started"},
               {"type": "item.completed", "item": {"type": "agent_message", "text": response}},
               {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 3,
                                                    "reasoning_output_tokens": 1}}]
    (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (folder / "response.json").write_text(response)
    (folder / "stderr.txt").write_text("")
    return folder


def test_requirements_notice_is_the_only_new_error_item_accepted(tmp_path):
    notice = ("Ignoring unknown `features` requirement `ultrafast_mode` from requirements layers: "
              "enterprise-managed requirements unrestricted-coding-assistant-codex (x)")
    usage = october2.codex_usage_allowing_notice(codex_stream(tmp_path / "a", [notice, notice]))
    assert usage["requirements_notices"] == 2 and usage["input_tokens"] == 10
    for bad in ("Selected model is at capacity.", "Ignoring unknown thing"):
        with pytest.raises(ValueError):
            october2.codex_usage_allowing_notice(codex_stream(tmp_path / bad[:8], [bad]))
    with pytest.raises(ValueError):  # Earlier recipes keep the strict validator.
        october2.codex_response_usage(codex_stream(tmp_path / "b", [notice]))


def test_october8_replications_route_new_draws_and_keep_earlier_ones():
    for d in "cdefgh":
        recipe = f"matched_all_views_{d}_vlm"
        spec = october2.RECIPES[recipe]
        assert spec["max_side"] == 460 and spec["views"] == "all" and spec["allow_requirements_notice"]
        assert october2.round_dir(recipe) == "october8"
    assert october2.round_dir("matched_all_views_a_vlm") == "october7r"
    assert october2.round_dir("matched_eight_draw_mean") == "october8"
    components, minimum = october2.MEANS["matched_eight_draw_mean"]
    assert len(components) == 8 and minimum == 6 and components[:2] == (
        "matched_all_views_a_vlm", "matched_all_views_b_vlm")
