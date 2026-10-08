"""The declared October 2 visual-language assays (docs/october2-plan.md); never uploads."""
from __future__ import annotations

import argparse
import base64
import re
import fcntl
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.evidence_visual_language import (
    response_usage as codex_response_usage, stderr_transport_usage, transport_events,
)
from soilgrain.metrics import emd_score
from soilgrain.retrieval_visual_language import file_record, json_bytes, save_once
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.visual_language import SCHEMA, audit_events, build_prompt, parse_prediction


CODEX = "/opt/homebrew/bin/codex"
CODEX_VERSION = "codex-cli 0.154.0"
CLAUDE = ("/Users/wmurray/Library/Application Support/Claude/claude-code/2.1.286/"
          "f2326db61802/claude.app/Contents/MacOS/claude")
CLAUDE_VERSION = "2.1.286 (Claude Code)"
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_SYSTEM = ("You estimate soil grain size distributions from attached photographs. "
                 "Follow the user's instructions and respond only through the requested JSON.")
PLAN = Path("docs/october2-plan.md")
ROUND2_PLAN = Path("docs/october2-round2-plan.md")
ROUND3_PLAN = Path("docs/october6-plan.md")
ROUND4_PLAN = Path("docs/october7-plan.md")
ROUND5_PLAN = Path("docs/october8-plan.md")
ORIGINAL = Path("artifacts/experiments/visual_language")
ORIGINAL_LEDGER_SHA256 = "dd4d650236ea87303248170937587129de988ef7f181e063af9abbaa1d6b86b3"
QUERIES = 34
ATTEMPTS = 2
RECIPES = {
    "rerun_a_vlm": {"provider": "codex", "reasoning": "low", "views": "camera", "max_side": 768},
    "rerun_b_vlm": {"provider": "codex", "reasoning": "low", "views": "camera", "max_side": 768},
    "all_views_vlm": {"provider": "codex", "reasoning": "low", "views": "all", "max_side": 768},
    "matched_resolution_vlm": {"provider": "codex", "reasoning": "low", "views": "camera",
                               "max_side": 460},
    "high_reasoning_vlm": {"provider": "codex", "reasoning": "high", "views": "camera",
                           "max_side": 768},
    "claude_vlm": {"provider": "claude", "reasoning": "low", "views": "camera", "max_side": 768},
}
# Round 2 (docs/october2-round2-plan.md): fresh draws of the all-photo winner, 150 mm crops, Claude.
# At 150 mm this one test photo shows a ruler and a company card, so wide recipes omit it.
NON_SOIL_AT_150MM = ("8d6d82eb17c2598a4348d8c3edee7cfadd6295c7d14ced895932a5b355d79b94",)
ROUND2 = {
    **{f"all_views_{d}_vlm": {"provider": "codex", "reasoning": "low", "views": "all",
                              "max_side": 768} for d in ("b", "c")},
    **{f"wide_all_views_{d}_vlm": {"provider": "codex", "reasoning": "low", "views": "all",
                                   "max_side": 768, "width_mm": 150,
                                   "exclude_sha256": NON_SOIL_AT_150MM} for d in ("a", "b", "c")},
    **{f"claude_all_views_{d}_vlm": {"provider": "claude", "reasoning": "low", "views": "all",
                                     "max_side": 768} for d in ("a", "b", "c")},
}
RECIPES.update(ROUND2)
# October 6 (docs/october6-plan.md): other random example panels, and two 100 mm side tiles per photo.
ROUND3 = {
    **{f"panel{seed}_all_views_vlm": {"provider": "codex", "reasoning": "low", "views": "all",
                                      "max_side": 768, "seed": seed, "round_dir": "october6"}
       for seed in (1, 2, 3, 4)},
    **{f"tiled_all_views_{d}_vlm": {"provider": "codex", "reasoning": "low", "views": "all",
                                    "max_side": 768, "tile_offsets_mm": (-50, 50),
                                    "round_dir": "october6"} for d in ("a", "b")},
}
RECIPES.update(ROUND3)
# October 7 (docs/october7-plan.md): match query resolution to the examples, and map every
# camera's mean exposure (examples included) to the training mean.
# Per-camera gains map each camera's mean 100 mm crop luminance to the training mean (129.615);
# computed once from unlabelled crops of every photo and frozen here.
EXPOSURE_GAINS = {"Motorola Edge": 1.0283, "Motorola Edge 60 Fusion": 0.8248, "Samsung A52": 0.9776,
                  "iPhone 14": 0.8333, "iPhone 16": 0.7799}
# The first October 7 run (artifacts/experiments/october7/) was rejected wholesale by a new client
# notice; the amended rerun accepts only that notice and writes to october7r/.
_BASE = {"provider": "codex", "reasoning": "low", "views": "all", "round_dir": "october7r",
         "allow_requirements_notice": True}
ROUND4 = {
    **{f"all_views_{d}_vlm": {**_BASE, "max_side": 768} for d in ("d", "e")},
    **{f"matched_all_views_{d}_vlm": {**_BASE, "max_side": 460} for d in ("a", "b")},
    **{f"exposure_all_views_{d}_vlm": {**_BASE, "max_side": 768, "exposure_gains": EXPOSURE_GAINS}
       for d in ("a", "b")},
    **{f"aligned_all_views_{d}_vlm": {**_BASE, "max_side": 460, "exposure_gains": EXPOSURE_GAINS}
       for d in ("a", "b")},
}
RECIPES.update(ROUND4)
# October 8 (docs/october8-plan.md): six fresh draws replicating the matched-resolution recipe.
ROUND5 = {f"matched_all_views_{d}_vlm": {**_BASE, "max_side": 460, "round_dir": "october8"}
          for d in ("c", "d", "e", "f", "g", "h")}
RECIPES.update(ROUND5)
# Coordinatewise means of independent draws: (components, minimum eligible components).
MEANS = {
    "all_views_draw_mean": (("all_views_vlm", "all_views_b_vlm", "all_views_c_vlm"), 2),
    "wide_draw_mean": (("wide_all_views_a_vlm", "wide_all_views_b_vlm", "wide_all_views_c_vlm"), 2),
    "all_scales_draw_mean": (("all_views_vlm", "all_views_b_vlm", "all_views_c_vlm",
                              "wide_all_views_a_vlm", "wide_all_views_b_vlm",
                              "wide_all_views_c_vlm"), 4),
    "claude_all_views_mean": (("claude_all_views_a_vlm", "claude_all_views_b_vlm",
                               "claude_all_views_c_vlm"), 2),
    # October 6: components may themselves be means; each component has equal weight.
    "panel_bagged_mean": (("all_views_draw_mean", "panel1_all_views_vlm", "panel2_all_views_vlm",
                           "panel3_all_views_vlm", "panel4_all_views_vlm"), 4),
    "tiled_mean": (("tiled_all_views_a_vlm", "tiled_all_views_b_vlm"), 2),
    "coverage_panel_mean": (("panel_bagged_mean", "tiled_mean"), 2),
}
ROUND3_MEANS = ("panel_bagged_mean", "tiled_mean", "coverage_panel_mean")
MEANS.update({
    "aligned_mean": (("aligned_all_views_a_vlm", "aligned_all_views_b_vlm"), 2),
    "matched_mean": (("matched_all_views_a_vlm", "matched_all_views_b_vlm"), 2),
    "exposure_mean": (("exposure_all_views_a_vlm", "exposure_all_views_b_vlm"), 2),
    "all_views_five_draw_mean": (("all_views_vlm", "all_views_b_vlm", "all_views_c_vlm",
                                  "all_views_d_vlm", "all_views_e_vlm"), 4),
})
ROUND4_MEANS = ("aligned_mean", "matched_mean", "exposure_mean", "all_views_five_draw_mean")
_NEW_MATCHED = tuple(f"matched_all_views_{d}_vlm" for d in ("c", "d", "e", "f", "g", "h"))
MEANS.update({
    "matched_new_six_mean": (_NEW_MATCHED, 5),
    "matched_eight_draw_mean": (("matched_all_views_a_vlm", "matched_all_views_b_vlm") + _NEW_MATCHED, 6),
    "matched_cd_mean": (("matched_all_views_c_vlm", "matched_all_views_d_vlm"), 2),
    "matched_ef_mean": (("matched_all_views_e_vlm", "matched_all_views_f_vlm"), 2),
    "matched_gh_mean": (("matched_all_views_g_vlm", "matched_all_views_h_vlm"), 2),
})
ROUND5_MEANS = ("matched_new_six_mean", "matched_eight_draw_mean", "matched_cd_mean",
                "matched_ef_mean", "matched_gh_mean")


def round_dir(name: str) -> str:
    if name in RECIPES:
        return RECIPES[name].get("round_dir", "october2")
    if name in ROUND5_MEANS:
        return "october8"
    if name in ROUND4_MEANS:
        return "october7r"
    return "october6" if name in ROUND3_MEANS else "october2"


def seeded_prompt(truth: pd.DataFrame, query_id: str, query_views: int,
                  seed: int) -> tuple[str, list[str]]:
    """``build_prompt`` with its example panel drawn from ``seed`` (seed 0 is the original)."""
    prompt, original = build_prompt(truth, query_id, query_views)
    if seed == 0:
        return prompt, original
    available = sorted(s for s in truth.sample_id.tolist() if s != query_id)
    examples = np.random.default_rng(seed).permutation(available)[:8].tolist()
    values = curve_array(truth.set_index("sample_id").loc[examples])
    lines = prompt.splitlines()
    expected = [f"Example image {i} mass percentages passing: " for i in range(1, 9)]
    if [line[:len(prefix)] for line, prefix in zip(lines[1:9], expected)] != expected:
        raise ValueError("Prompt example lines changed")
    lines[1:9] = [f"{prefix}{row.tolist()}" for prefix, row in zip(expected, values)]
    return "\n".join(lines), examples
UNKNOWN_INPUT_TOKENS = 40_000
UNKNOWN_OUTPUT_TOKENS = {"low": 2_000, "high": 20_000, "claude": 4_000}
BUDGETS = {
    "low": {"input_tokens": 2_000_000, "output_tokens": 60_000, "timeout": 300, "workers": 2},
    "high": {"input_tokens": 2_000_000, "output_tokens": 600_000, "timeout": 900, "workers": 2},
    "claude": {"input_tokens": 2_000_000, "output_tokens": 150_000, "timeout": 300, "workers": 3,
               "usd": 60.0, "usd_per_call": 2.0},
}


def budget(recipe: str) -> dict:
    spec = RECIPES[recipe]
    return BUDGETS["claude" if spec["provider"] == "claude" else spec["reasoning"]]


def prepare_crop(path: Path, camera: pd.Series, output: Path, max_side: int,
                 width_mm: int = 100, offset_mm: int = 0, gain: float | None = None) -> None:
    """A calibrated center crop; 100 mm at ``max_side=768`` is byte-identical to the original."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    ppm = float(camera["ppm"]) * max(image.size) / max(camera["width"], camera["height"])
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("Invalid physical calibration")
    side = round(width_mm * ppm)
    if not 1 <= side <= min(image.size):
        raise ValueError(f"{width_mm}mm crop does not fit")
    left, top = (image.width - side) // 2, (image.height - side) // 2
    if offset_mm:  # Shift along the long side of the photograph.
        if image.width >= image.height:
            left += round(offset_mm * ppm)
        else:
            top += round(offset_mm * ppm)
        if left < 0 or top < 0 or left + side > image.width or top + side > image.height:
            raise ValueError("Tile does not fit")
    crop = image.crop((left, top, left + side, top + side))
    if side > max_side:
        crop = crop.resize((max_side, max_side), Image.Resampling.LANCZOS)
    if gain is not None:  # Camera exposure gain after resizing: round half to even, clip to 0-255.
        scaled = np.clip(np.rint(np.asarray(crop, dtype=float) * gain), 0, 255).astype(np.uint8)
        crop = Image.fromarray(scaled, mode="RGB")
    crop.info.clear()
    crop.save(output, format="PNG")


def request_context(recipe: str, truth: pd.DataFrame, photos: pd.DataFrame, query_id: str):
    spec = RECIPES[recipe]
    index = photos.sort_values(["camera", "path"])
    views = index.loc[index.sample_id == query_id]
    if spec["views"] == "camera":
        views = views.groupby("camera", sort=True).head(1)
    if spec.get("exclude_sha256"):
        keep = [file_record(Path(path))["sha256"] not in spec["exclude_sha256"] for path in views.path]
        views = views.loc[keep]
    if views.empty:
        raise ValueError("Query requires at least one photo")
    offsets = spec.get("tile_offsets_mm")
    query_images = len(views) * (len(offsets) if offsets else 1)
    prompt, examples = seeded_prompt(truth, query_id, query_images, spec.get("seed", 0))
    if offsets:
        original = "Each attached image is a 100 mm square center crop."
        if prompt.count(original) != 1:
            raise ValueError("Prompt crop sentence changed")
        prompt = prompt.replace(original, "Each attached image is a 100 mm square crop.")
    width = spec.get("width_mm", 100)
    if width != 100:
        original = "Each attached image is a 100 mm square center crop."
        if prompt.count(original) != 1 or prompt.count("100 mm") != 1:
            raise ValueError("Prompt crop-width sentence changed")
        prompt = prompt.replace(original, f"Each attached image is a {width} mm square center crop.")
    if query_id in examples:
        raise ValueError("Held-out soil cannot be an example")
    query_rows = [row for _, row in views.iterrows()]
    if offsets:
        query_rows = [pd.concat([row, pd.Series({"offset_mm": offset})])
                      for row in query_rows for offset in offsets]
    rows = [index.loc[(index.split == "train") & (index.sample_id == sample_id)].iloc[0]
            for sample_id in examples] + query_rows
    selection = {"examples": examples,
                 "image_sources": [{**file_record(Path(row.path)), "camera": row.camera,
                                    **({"offset_mm": int(row["offset_mm"])} if "offset_mm" in row
                                       else {})} for row in rows]}
    return prompt, selection, rows


def render_images(recipe: str, rows, cameras: pd.DataFrame, folder: Path) -> list[Path]:
    images = []
    for number, row in enumerate(rows, 1):
        output = folder / f"image_{number:02d}.png"
        gains = RECIPES[recipe].get("exposure_gains")
        prepare_crop(Path(row.path), cameras.loc[row.camera], output, RECIPES[recipe]["max_side"],
                     RECIPES[recipe].get("width_mm", 100), int(row.get("offset_mm", 0)),
                     gains[row.camera] if gains else None)
        images.append(output)
    return images


def codex_command(folder: Path, images: list[Path], schema: Path, reasoning: str) -> list[str]:
    command = [
        CODEX, "exec", "--ignore-user-config", "--ignore-rules",
        "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--cd", str(folder),
    ]
    for feature in ("apps", "plugins", "hooks", "memories", "multi_agent", "shell_tool",
                    "unified_exec", "view_image", "image_generation", "code_mode_host", "skill_search"):
        command += ["--disable", feature]
    for config in ('web_search="disabled"', "project_doc_max_bytes=0",
                   'model="gpt-6-astra"', f'model_reasoning_effort="{reasoning}"'):
        command += ["-c", config]
    for path in images:
        command += ["--image", str(path)]
    return command + ["--output-schema", str(schema), "--output-last-message",
                      str(folder / "response.json"), "--json", "-"]


def claude_command(reasoning: str) -> list[str]:
    return [
        CLAUDE, "-p", "--safe-mode", "--tools", "", "--strict-mcp-config",
        "--disable-slash-commands", "--setting-sources", "", "--system-prompt", CLAUDE_SYSTEM,
        "--no-session-persistence", "--input-format", "stream-json",
        "--output-format", "stream-json", "--verbose", "--model", CLAUDE_MODEL,
        "--effort", reasoning, "--json-schema", json.dumps(SCHEMA),
        "--max-budget-usd", str(BUDGETS["claude"]["usd_per_call"]),
    ]


def claude_message(images: list[Path], prompt: str) -> str:
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(path.read_bytes()).decode()}}
               for path in images]
    content.append({"type": "text", "text": prompt})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def claude_response_usage(folder: Path) -> dict:
    """Validate an isolated Claude stream and return its usage; writes nothing."""
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()
              if line.strip()]
    inits = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    results = [e for e in events if e.get("type") == "result"]
    if len(inits) != 1 or len(results) != 1:
        raise ValueError("Require one init and one result event")
    init, result = inits[0], results[0]
    if (init.get("model") != CLAUDE_MODEL or init.get("tools") != ["StructuredOutput"]
            or init.get("mcp_servers") != []):
        raise ValueError("Claude session was not isolated to the declared model and schema tool")
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_use" and block.get("name") != "StructuredOutput":
                raise ValueError(f"Unexpected tool use: {block.get('name')}")
    models = set(result.get("modelUsage", {}))
    if (result.get("subtype") != "success" or result.get("is_error") or models != {CLAUDE_MODEL}
            or not isinstance(result.get("structured_output"), dict)):
        raise ValueError("Claude request did not succeed with the declared model")
    response = (folder / "response.json").read_text()
    if json.loads(response) != result["structured_output"]:
        raise ValueError("Response file differs from the structured output")
    parse_prediction(response)
    usage = result["usage"]
    input_tokens = sum(int(usage.get(key, 0)) for key in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    thinking = int(usage.get("output_tokens_details", {}).get("thinking_tokens", 0))
    return {"input_tokens": input_tokens, "output_tokens": int(usage["output_tokens"]),
            "reasoning_output_tokens": thinking, "cost_usd": float(result["total_cost_usd"])}


class LaunchError(RuntimeError):
    """The client could not be started; no model request was made."""


def _events(folder: Path) -> list[dict]:
    events = []
    for line in (folder / "events.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # The raw log is preserved; validation rejects a broken stream.
        if isinstance(event, dict):
            events.append(event)
    return events


def extract_claude_response(folder: Path) -> None:
    """Persist the stream's structured output, if any, before validating it."""
    results = [e for e in _events(folder) if e.get("type") == "result"]
    if len(results) == 1 and isinstance(results[0].get("structured_output"), dict):
        save_once(folder / "response.json", json.dumps(results[0]["structured_output"]).encode())


# A content-independent client notice about managed account settings that Codex 0.154.0 does not
# recognize; accepted only for recipes declared with ``allow_requirements_notice``.
REQUIREMENTS_NOTICE = re.compile(
    r"Ignoring unknown `features` requirement `[a-z_]+` from requirements layers: "
    r"enterprise-managed requirements .+")


def codex_usage_allowing_notice(folder: Path) -> dict:
    """``evidence_visual_language.response_usage`` after removing requirements notices."""
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    kept = [event for event in events
            if not (event.get("type") == "item.completed"
                    and event.get("item", {}).get("type") == "error"
                    and REQUIREMENTS_NOTICE.fullmatch(event["item"].get("message", "")))]
    filtered, transport = transport_events(kept)
    usage = audit_events(filtered)
    messages = [event["item"]["text"] for event in filtered
                if event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "agent_message"]
    response = (folder / "response.json").read_text()
    if len(messages) != 1 or messages[0].strip() != response.strip():
        raise ValueError("Require exactly one final message matching the response file")
    parse_prediction(response)
    return {**usage, "transport_events": transport, "requirements_notices": len(events) - len(kept),
            **stderr_transport_usage(folder / "stderr.txt")}


def validated(recipe: str, folder: Path) -> dict:
    if RECIPES[recipe]["provider"] == "codex":
        usage = (codex_usage_allowing_notice(folder) if RECIPES[recipe].get("allow_requirements_notice")
                 else codex_response_usage(folder))
    else:
        extract_claude_response(folder)
        usage = claude_response_usage(folder)
    return {**usage, "curve": parse_prediction((folder / "response.json").read_text()).tolist()}


def charged_usage(recipe: str, folder: Path) -> dict:
    """Best-effort spend of a rejected attempt; unknown spend is charged at a ceiling."""
    events = _events(folder) if (folder / "events.jsonl").exists() else []
    if RECIPES[recipe]["provider"] == "codex":
        done = [e for e in events if e.get("type") == "turn.completed" and "usage" in e]
        if done:
            usage = done[-1]["usage"]
            return {key: int(usage.get(key, 0)) for key in
                    ("input_tokens", "output_tokens", "reasoning_output_tokens")}
        return {"input_tokens": UNKNOWN_INPUT_TOKENS,
                "output_tokens": UNKNOWN_OUTPUT_TOKENS[RECIPES[recipe]["reasoning"]]}
    results = [e for e in events if e.get("type") == "result" and "usage" in e]
    if results:
        usage = results[-1]["usage"]
        return {"input_tokens": sum(int(usage.get(key, 0)) for key in (
                    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "cost_usd": float(results[-1].get("total_cost_usd", BUDGETS["claude"]["usd_per_call"]))}
    return {"input_tokens": UNKNOWN_INPUT_TOKENS, "output_tokens": UNKNOWN_OUTPUT_TOKENS["claude"],
            "cost_usd": BUDGETS["claude"]["usd_per_call"]}


def dispatch(recipe: str, folder: Path, images: list[Path], prompt: str, schema: Path,
             run=subprocess.run) -> dict:
    """Run one isolated request in ``folder``; return validated usage and the curve."""
    spec = RECIPES[recipe]
    timeout = budget(recipe)["timeout"]
    with (folder / "events.jsonl").open("x") as stdout, (folder / "stderr.txt").open("x") as stderr:
        # A separate session keeps a terminal Ctrl-C from killing in-flight requests.
        try:
            if spec["provider"] == "codex":
                result = run(codex_command(folder, images, schema, spec["reasoning"]), input=prompt,
                             text=True, stdout=stdout, stderr=stderr, timeout=timeout,
                             start_new_session=True)
            else:
                result = run(claude_command(spec["reasoning"]), input=claude_message(images, prompt),
                             text=True, stdout=stdout, stderr=stderr, timeout=timeout, cwd=folder,
                             start_new_session=True)
        except OSError as error:
            raise LaunchError(f"Client could not start: {error}") from error
    if result.returncode:
        if spec["provider"] == "claude":
            extract_claude_response(folder)
        raise RuntimeError(f"Model client returned {result.returncode}")
    return validated(recipe, folder)


def totals(records: list[dict]) -> dict:
    keys = ("input_tokens", "output_tokens", "reasoning_output_tokens", "cost_usd")
    return {key: sum(record.get(key, 0) for record in records) for key in keys}


def load_inputs(config_path: str):
    cfg = load_config(config_path)
    truth, sample, photos, _components, sources = load_search_inputs(config_path)
    sources += _verified_sources([{"path": str(ORIGINAL / "requests.json"),
                                   "sha256": ORIGINAL_LEDGER_SHA256}])
    sources.append(file_record(PLAN))
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    return cfg, truth, sample, photos["spectral"], cameras, sources


def verify_reruns_match_original(truth, sample, photos, cameras, folder: Path) -> None:
    """The rerun recipes must rebuild the original prompts and images byte-for-byte."""
    ledger = json.loads((ORIGINAL / "requests.json").read_text())
    records = {record["sample_id"]: record for record in ledger["requests"]}
    for query_id in truth.sample_id.tolist() + sample.sample_id.tolist():
        prompt, selection, rows = request_context("rerun_a_vlm", truth, photos, query_id)
        target = folder / query_id.replace(" ", "_")
        target.mkdir(parents=True)
        images = render_images("rerun_a_vlm", rows, cameras, target)
        record = records[query_id]
        if (sha256(prompt.encode()).hexdigest() != record["prompt_sha256"]
                or selection["examples"] != record["examples"]
                or [file_record(path)["sha256"] for path in images] != record["image_sha256"]):
            raise ValueError(f"Rerun context differs from the original request: {query_id}")


ALL_VIEWS = Path("artifacts/experiments/october2/all_views_vlm")
ALL_VIEWS_SUMMARY_SHA256 = "6e076cfd756a93e7ffab87763881767790b2d24ee8931734e063c4b9d62fa424"


def verify_matches_all_views(recipe, truth, sample, photos, cameras, folder: Path) -> None:
    """All-photo 100 mm redraws must rebuild the winning requests' prompts and images exactly."""
    _verified_sources([{"path": str(ALL_VIEWS / "summary.json"), "sha256": ALL_VIEWS_SUMMARY_SHA256}])
    records = {}
    for path in sorted((ALL_VIEWS / "records").glob("*_attempt_1.json")):
        record = json.loads(path.read_text())
        _verified_sources(record["raw_files"])
        records[record["sample_id"]] = record
    for query_id in truth.sample_id.tolist() + sample.sample_id.tolist():
        prompt, selection, rows = request_context(recipe, truth, photos, query_id)
        target = folder / query_id.replace(" ", "_")
        target.mkdir(parents=True)
        images = render_images(recipe, rows, cameras, target)
        record = records[query_id]
        if (record["state"] != "complete"
                or sha256(prompt.encode()).hexdigest() != record["prompt_sha256"]
                or selection["examples"] != record["examples"]
                or [file_record(path)["sha256"] for path in images] != record["image_sha256"]):
            raise ValueError(f"Redraw context differs from the all-photo winner: {query_id}")


def runtime_identity(recipe: str) -> dict:
    paths = [Path(__file__).resolve(), PLAN.resolve()]
    if recipe in ROUND2:
        paths.append(ROUND2_PLAN.resolve())
    if recipe in ROUND3:
        paths.append(ROUND3_PLAN.resolve())
    if recipe in ROUND4:
        paths.append(ROUND4_PLAN.resolve())
    if recipe in ROUND5:
        paths.append(ROUND5_PLAN.resolve())
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    code = [file_record(path) for path in paths]
    for path, record in zip(paths, code, strict=True):
        committed = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(root)}"])
        if sha256(committed).hexdigest() != record["sha256"]:
            raise ValueError("Commit the producing code and declaration before any model request")
    if RECIPES[recipe]["provider"] == "codex":
        version = subprocess.check_output([CODEX, "--version"], text=True).strip()
        expected = CODEX_VERSION
    else:
        version = subprocess.check_output([CLAUDE, "--version"], text=True).strip()
        expected = CLAUDE_VERSION
    if version != expected:
        raise ValueError(f"Client version {version} differs from the declaration")
    return {"recipe": recipe, **RECIPES[recipe], "budget": budget(recipe),
            "client_version": version,
            "producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "code_sources": code}


def run_recipe(recipe: str, config_path: str = "configs/data.yaml") -> dict:
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    cfg, truth, sample, photos, cameras, sources = load_inputs(config_path)
    _verified_sources(sources)
    query_ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    if len(truth) != 24 or len(query_ids) != QUERIES or len(set(query_ids)) != QUERIES:
        raise ValueError("The declaration requires 24 distinct holdouts and ten distinct test soils")
    identity = runtime_identity(recipe)
    spec = RECIPES[recipe]
    if (spec["max_side"] == 768 and spec.get("width_mm", 100) == 100
            and spec.get("seed", 0) == 0 and not spec.get("tile_offsets_mm")
            and not spec.get("exposure_gains")):
        verify = (verify_reruns_match_original if spec["views"] == "camera"
                  else lambda *args: verify_matches_all_views(recipe, *args))
        with TemporaryDirectory(prefix="soilgrain-october2-") as temporary:
            verify(truth, sample, photos, cameras, Path(temporary))
    output = (cfg.artifacts_dir / "experiments" / round_dir(recipe) / recipe).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = output / "identity.json"
        save_once(identity_path, json_bytes({**identity, "sources": sources, "query_ids": query_ids}))
        return _run(output, recipe, truth, sample, photos, cameras, query_ids)


def _run(output: Path, recipe: str, truth, sample, photos, cameras, query_ids,
         run=subprocess.run, progress=print) -> dict:
    limits = budget(recipe)
    schema = output / "schema.json"
    save_once(schema, json_bytes(SCHEMA))
    lock = threading.Lock()
    stop = threading.Event()
    completed = [json.loads(path.read_text()) for path in sorted((output / "records").glob("*.json"))]

    def within_budget() -> bool:
        spent = totals(completed)
        return (spent["input_tokens"] < limits["input_tokens"]
                and spent["output_tokens"] < limits["output_tokens"]
                and spent["cost_usd"] < limits.get("usd", float("inf")))

    def finish(record: dict, record_path: Path, folder: Path) -> dict:
        record["raw_files"] = [file_record(path) for path in sorted(folder.iterdir()) if path.is_file()]
        save_once(record_path, json_bytes(record))
        with lock:
            completed.append(record)
            progress(f"{recipe} {record_path.stem} {record['state']}; {totals(completed)}", flush=True)
        return record

    def solve(number: int) -> dict | None:
        query_id = query_ids[number]
        prompt, selection, rows = request_context(recipe, truth, photos, query_id)
        query = output / "requests" / f"query_{number:02d}"
        for attempt in range(1, ATTEMPTS + 1):
            folder = query / f"attempt_{attempt}"
            name = f"query_{number:02d}_attempt_{attempt}"
            record_path = output / "records" / f"{name}.json"
            base = {"sample_id": query_id, "split": "train" if number < 24 else "test",
                    "attempt": attempt, **selection, "folder": str(folder),
                    "prompt_sha256": sha256(prompt.encode()).hexdigest()}
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["state"] == "complete":
                    _verified_sources(record["raw_files"])
                    return record
                continue
            if (folder / "events.jsonl").exists():
                # Dispatched before an interruption: use a valid response, never re-request it.
                record = {**base, "image_sha256": json.loads(
                    (output / "receipts" / f"{name}.json").read_text())["image_sha256"]}
                try:
                    record.update(validated(recipe, folder), state="complete", recovered=True)
                except Exception as error:
                    record.update(charged_usage(recipe, folder), state="failed", recovered=True,
                                  error_type=type(error).__name__, error=str(error))
                if finish(record, record_path, folder)["state"] == "complete":
                    return record
                continue
            with lock:
                if stop.is_set():
                    return None
                if not within_budget():
                    stop.set()
                    raise ValueError("Declared aggregate token/cost budget reached")
            folder.mkdir(parents=True, exist_ok=True)  # Undispatched folders are rebuilt.
            images = render_images(recipe, rows, cameras, folder)
            save_once(folder / "prompt.txt", prompt.encode())
            record = {**base, "image_sha256": [file_record(path)["sha256"] for path in images]}
            save_once(output / "receipts" / f"{name}.json", json_bytes(record))
            try:
                record.update(dispatch(recipe, folder, images, prompt, schema, run), state="complete")
            except LaunchError:
                (folder / "events.jsonl").unlink(missing_ok=True)  # Nothing was sent.
                (folder / "stderr.txt").unlink(missing_ok=True)
                stop.set()
                raise
            except Exception as error:  # Objective failure: preserved, at most one more dispatch.
                record.update(charged_usage(recipe, folder), state="failed",
                              error_type=type(error).__name__, error=str(error))
            if finish(record, record_path, folder)["state"] == "complete":
                return record
        stop.set()  # The recipe is ineligible; spend nothing more on it.
        return None

    with ThreadPoolExecutor(max_workers=limits["workers"]) as pool:
        futures = [pool.submit(solve, number) for number in range(len(query_ids))]
        try:
            while not all(future.done() for future in futures):
                wait(futures, timeout=1)
        except KeyboardInterrupt:
            stop.set()  # In-flight requests finish and are recorded; nothing new starts.
            for future in futures:
                future.cancel()
            raise
        results = [future.result() for future in futures]
    spent = totals(completed)
    if any(result is None for result in results):
        failure = {"eligible": False, "experiment": recipe, **spent,
                   "missing": [query_ids[i] for i, result in enumerate(results) if result is None]}
        save_once(output / "failure.json", json_bytes(failure))
        raise ValueError(f"{recipe} is ineligible: a query lacks a valid attempt")
    return finalize(output, recipe, truth, sample, np.asarray([r["curve"] for r in results]),
                    {**spent, "attempts": len(completed),
                     "failed_attempts": sum(r["state"] != "complete" for r in completed)})


def finalize(output: Path, recipe: str, truth: pd.DataFrame, sample: pd.DataFrame,
             curves: np.ndarray, extra: dict) -> dict:
    if curves.shape != (len(truth) + len(sample), 11):
        raise ValueError("Require one curve per holdout and test soil")
    oof = curves[:len(truth)]
    frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, "sample_id", truth.sample_id.tolist())
    frame["emd"] = [emd_score(actual, predicted)
                    for actual, predicted in zip(curve_array(truth), oof, strict=True)]
    candidate = sample.copy()
    candidate[ordered_grain_columns(sample)] = curves[len(truth):]
    validate_submission(candidate, sample)
    oof_path, submission_path = output / "oof.csv", output / f"{recipe}.csv"
    save_once(oof_path, frame.to_csv(index=False).encode())
    save_once(submission_path, candidate.to_csv(index=False).encode())
    summary = {"experiment": recipe, "eligible": True,
               "loo_emd": emd_score(curve_array(truth), oof), "heldout_soils": len(truth),
               "test_soils": len(sample), **extra, "oof": file_record(oof_path),
               "submission": file_record(submission_path)}
    save_once(output / "summary.json", json_bytes(summary))
    return summary


def draw_mean(config_path: str = "configs/data.yaml") -> dict:
    """Coordinatewise mean of the saved original VLM and every eligible rerun draw."""
    cfg, truth, sample, _photos, _cameras, sources = load_inputs(config_path)
    _verified_sources(sources)
    base = cfg.artifacts_dir / "experiments"
    original_summary = json.loads((base / "visual_language" / "summary.json").read_text())
    paths = [(Path(original_summary["oof"]["path"]), Path(original_summary["submission"]["path"]))]
    pinned = [original_summary["oof"], original_summary["submission"]]
    for recipe in ("rerun_a_vlm", "rerun_b_vlm"):
        summary_path = base / "october2" / recipe / "summary.json"
        failure_path = base / "october2" / recipe / "failure.json"
        if summary_path.exists() == failure_path.exists():
            raise ValueError(f"{recipe} must have finished as exactly one of eligible or failed")
        if failure_path.exists():
            if json.loads(failure_path.read_text())["eligible"] is not False:
                raise ValueError(f"{recipe} failure record is inconsistent")
            continue
        summary = json.loads(summary_path.read_text())
        if summary["experiment"] != recipe or summary["eligible"] is not True:
            raise ValueError(f"{recipe} summary is inconsistent")
        paths.append((Path(summary["oof"]["path"]), Path(summary["submission"]["path"])))
        pinned += [summary["oof"], summary["submission"]]
    if len(paths) < 2:
        raise ValueError("Require at least one eligible rerun draw")
    _verified_sources(pinned)
    curves = []
    for oof_path, submission_path in paths:
        oof, test = pd.read_csv(oof_path), pd.read_csv(submission_path)
        if (sorted(oof.sample_id) != sorted(truth.sample_id)
                or sorted(test.sample_id) != sorted(sample.sample_id)):
            raise ValueError("Each draw requires exactly the holdout and test soils")
        validate_cumulative_curves(oof)
        validate_cumulative_curves(test)
        curves.append(np.vstack([curve_array(oof.set_index("sample_id").loc[truth.sample_id]),
                                 curve_array(test.set_index("sample_id").loc[sample.sample_id])]))
    output = base / "october2" / "vlm_draw_mean"
    output.mkdir(parents=True, exist_ok=True)
    return finalize(output, "vlm_draw_mean", truth, sample, np.mean(curves, axis=0),
                    {"draws": len(paths), "components": pinned, "producing_commit":
                     subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})


def recipe_mean(name: str, config_path: str = "configs/data.yaml") -> dict:
    """Coordinatewise mean of the eligible draws declared for ``name`` in ``MEANS``."""
    components, minimum = MEANS[name]
    cfg, truth, sample, _photos, _cameras, sources = load_inputs(config_path)
    _verified_sources(sources)
    base = cfg.artifacts_dir / "experiments"
    curves, pinned, excluded = [], [], []
    for recipe in components:
        folder = base / round_dir(recipe) / recipe
        summary_path, failure_path = folder / "summary.json", folder / "failure.json"
        if summary_path.exists() == failure_path.exists():
            raise ValueError(f"{recipe} must have finished as exactly one of eligible or failed")
        if failure_path.exists():
            if json.loads(failure_path.read_text())["eligible"] is not False:
                raise ValueError(f"{recipe} failure record is inconsistent")
            excluded.append(recipe)
            continue
        summary = json.loads(summary_path.read_text())
        if summary["experiment"] != recipe or summary["eligible"] is not True:
            raise ValueError(f"{recipe} summary is inconsistent")
        _verified_sources([summary["oof"], summary["submission"]])
        oof, test = pd.read_csv(summary["oof"]["path"]), pd.read_csv(summary["submission"]["path"])
        if (sorted(oof.sample_id) != sorted(truth.sample_id)
                or sorted(test.sample_id) != sorted(sample.sample_id)):
            raise ValueError("Each draw requires exactly the holdout and test soils")
        validate_cumulative_curves(oof)
        validate_cumulative_curves(test)
        curves.append(np.vstack([curve_array(oof.set_index("sample_id").loc[truth.sample_id]),
                                 curve_array(test.set_index("sample_id").loc[sample.sample_id])]))
        pinned += [summary["oof"], summary["submission"]]
    output = base / round_dir(name) / name
    if len(curves) < minimum:
        output.mkdir(parents=True, exist_ok=True)
        save_once(output / "failure.json", json_bytes({"eligible": False, "experiment": name,
                                                       "excluded": excluded}))
        raise ValueError(f"{name} needs at least {minimum} eligible draws")
    output.mkdir(parents=True, exist_ok=True)
    return finalize(output, name, truth, sample, np.mean(curves, axis=0),
                    {"draws": len(curves), "excluded": excluded, "components": pinned,
                     "producing_commit": subprocess.check_output(
                         ["git", "rev-parse", "HEAD"], text=True).strip()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=[*RECIPES, "vlm_draw_mean", *MEANS], required=True)
    parser.add_argument("--config-path", default="configs/data.yaml")
    arguments = parser.parse_args()
    if arguments.recipe == "vlm_draw_mean":
        print(json.dumps(draw_mean(arguments.config_path), indent=2))
    elif arguments.recipe in MEANS:
        print(json.dumps(recipe_mean(arguments.recipe, arguments.config_path), indent=2))
    else:
        print(json.dumps(run_recipe(arguments.recipe, arguments.config_path), indent=2))


if __name__ == "__main__":
    main()
