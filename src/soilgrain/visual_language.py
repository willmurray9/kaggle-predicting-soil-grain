"""A fixed, isolated visual-language assay; never uploads to Kaggle."""
from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.targets import curve_array


SCHEMA = {
    "type": "object", "properties": {"cdf": {
        "type": "array", "items": {"type": "number"}, "minItems": 11, "maxItems": 11,
    }}, "required": ["cdf"], "additionalProperties": False,
}


def build_prompt(truth: pd.DataFrame, query_id: str, query_views: int) -> tuple[str, list[str]]:
    available = sorted(s for s in truth.sample_id.tolist() if s != query_id)
    if len(available) < 8 or len(set(available)) != len(available) or query_views < 1:
        raise ValueError("Require eight distinct training examples and a query view")
    examples = np.random.default_rng(0).permutation(available)[:8].tolist()
    values = curve_array(truth.set_index("sample_id").loc[examples])
    prompt = (
        "Estimate a soil sample's laboratory mass-based cumulative grain size distribution "
        "from calibrated photographs. Each attached image is a 100 mm square center crop. "
        "Photographs can differ in camera, resolution, color rendering and arrangement. "
        "The first 8 images are independent labeled examples, one photo per soil. "
        f"The remaining {query_views} query images show the SAME new soil from different views. "
        "Use the examples and visual soil texture/type to infer its distribution, including "
        "unresolved fine material. Percentages describe dry bulk MASS passing each diameter, "
        "not visible particle counts or surface area. Never identify the soil or retrieve data. "
        "Use only attached images and these examples; do not use tools or access files. "
        f"Diameters in mm, in order: {list(SUPPORT_DIAMETERS)}.\n"
    )
    for i, row in enumerate(values, 1):
        prompt += f"Example image {i} mass percentages passing: {row.tolist()}\n"
    prompt += (
        'Return only {"cdf": [eleven numeric percentages]} for the query soil. '
        "Values must be finite, between 0 and 100, nondecreasing, with the last exactly 100. "
        "Give one best estimate, without commentary."
    )
    return prompt, examples


def parse_prediction(text: str) -> np.ndarray:
    if len(text) > 2048:
        raise ValueError("Response exceeds the declared final-text ceiling")
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"cdf"}:
        raise ValueError("Response must contain only cdf")
    if not isinstance(value["cdf"], list) or any(type(x) not in (int, float) for x in value["cdf"]):
        raise ValueError("CDF must contain numbers")
    curve = np.asarray(value["cdf"], dtype=float)
    if (curve.shape != (11,) or not np.isfinite(curve).all()
            or np.any((curve < 0) | (curve > 100)) or np.any(np.diff(curve) < 0)
            or curve[-1] != 100):
        raise ValueError("Response is not a valid eleven-point cumulative curve")
    return curve


def audit_events(events: list[dict]) -> dict[str, int]:
    completed = [e for e in events if e.get("type") == "turn.completed"]
    if len(completed) != 1 or "usage" not in completed[0]:
        raise ValueError("Require one completed turn with recorded usage")
    for event in events:
        item = event.get("item", {})
        kind = item.get("type")
        if kind == "error" and item.get("message", "").startswith("Code Mode is unavailable"):
            continue  # The disabled executor fails closed; it is not a tool invocation.
        if kind not in (None, "agent_message", "reasoning"):
            raise ValueError(f"Unexpected event/tool use: {kind}")
        if event.get("type") in ("error", "turn.failed"):
            raise ValueError("Model request failed")
    usage = completed[0]["usage"]
    result = {key: int(usage[key]) for key in ("input_tokens", "output_tokens", "reasoning_output_tokens")}
    if min(result.values()) < 0 or result["reasoning_output_tokens"] > result["output_tokens"]:
        raise ValueError("Unexpected token accounting; output total must include reasoning")
    return result


def audit_cached_record(record: dict) -> None:
    folder, response = Path(record["folder"]), Path(record["response_path"])
    images = sorted(folder.glob("image_*.png"))
    if (sha256(response.read_bytes()).hexdigest() != record["response_sha256"]
            or sha256((folder / "prompt.txt").read_bytes()).hexdigest() != record["prompt_sha256"]
            or [sha256(p.read_bytes()).hexdigest() for p in images] != record["image_sha256"]
            or not np.array_equal(parse_prediction(response.read_text()), record["curve"])):
        raise ValueError("Recorded model response, prompt or image changed")


def cli_command(folder: Path, images: list[Path], schema: Path) -> list[str]:
    command = [
        "/opt/homebrew/bin/codex", "exec", "--ignore-user-config", "--ignore-rules",
        "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--cd", str(folder),
    ]
    for feature in ("apps", "plugins", "hooks", "memories", "multi_agent", "shell_tool",
                    "unified_exec", "view_image", "image_generation", "code_mode_host", "skill_search"):
        command += ["--disable", feature]
    for config in ('web_search="disabled"', "project_doc_max_bytes=0",
                   'model="gpt-6-astra"', 'model_reasoning_effort="low"'):
        command += ["-c", config]
    for path in images:
        command += ["--image", str(path)]
    return command + ["--output-schema", str(schema), "--output-last-message",
                      str(folder / "response.json"), "--json", "-"]


def prepare_crop(path: Path, camera: pd.Series, output: Path) -> None:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    ppm = float(camera["ppm"]) * max(image.size) / max(camera["width"], camera["height"])
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("Invalid physical calibration")
    side = round(100 * ppm)
    if not 1 <= side <= min(image.size):
        raise ValueError("100mm crop does not fit")
    left, top = (image.width - side) // 2, (image.height - side) // 2
    crop = image.crop((left, top, left + side, top + side))
    if side > 768:
        crop = crop.resize((768, 768), Image.Resampling.LANCZOS)
    crop.info.clear()
    crop.save(output, format="PNG")


def run_visual_language(config_path: str = "configs/data.yaml", *, include_test: bool = False) -> dict:
    from soilgrain.config import load_config
    from soilgrain.io import write_json
    from soilgrain.metrics import emd_score
    from soilgrain.search_inputs import load_search_inputs
    from soilgrain.submission import validate_submission
    from soilgrain.targets import ordered_grain_columns

    cfg = load_config(config_path)
    truth, sample, photos, _components, sources = load_search_inputs(config_path)
    output = cfg.artifacts_dir / "experiments" / "visual_language"
    output.mkdir(parents=True, exist_ok=True)
    schema = (output / "schema.json").resolve()
    if not schema.exists():
        write_json(SCHEMA, schema)
    elif json.loads(schema.read_text()) != SCHEMA:
        raise ValueError("Recorded schema differs")
    index = photos["spectral"].sort_values(["camera", "path"])
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    ledger_path = output / "requests.json"
    client_version = subprocess.check_output(["/opt/homebrew/bin/codex", "--version"], text=True).strip()
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "smoke_requests": 2, "smoke_input_tokens": 26392, "smoke_output_tokens": 102,
        "smoke_reasoning_output_tokens": 50,
        "requests": [], "model": "gpt-6-astra", "reasoning": "low",
        "client_version": client_version,
        "producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
    }
    if ledger["code_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Model runner changed; cannot mix recipes across resumed requests")
    if ledger["client_version"] != client_version:
        raise ValueError("Model client changed; cannot mix client versions across resumed requests")
    if any(r["state"] != "complete" for r in ledger["requests"]):
        raise ValueError("Resolve unfinished/failed request explicitly before continuing; no automatic retry")
    query_ids = truth.sample_id.tolist() + (sample.sample_id.tolist() if include_test else [])
    for number, query_id in enumerate(query_ids):
        existing = next((r for r in ledger["requests"] if r["sample_id"] == query_id), None)
        if existing:
            audit_cached_record(existing)
            continue
        totals = {k: ledger[f"smoke_{k}"] + sum(r[k] for r in ledger["requests"])
                  for k in ("input_tokens", "output_tokens")}
        if len(ledger["requests"]) + ledger["smoke_requests"] >= 36 or totals["input_tokens"] >= 1000000 or totals["output_tokens"] >= 50000:
            raise ValueError("Declared model request/token budget reached")
        # Anonymous folder names and image filenames provide no soil identifiers.
        folder = (output / "requests" / f"query_{number:02d}").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        views = index.loc[index.sample_id == query_id].groupby("camera", sort=True).head(1)
        prompt, examples = build_prompt(truth, query_id, len(views))
        selected = [index.loc[(index.split == "train") & (index.sample_id == s)].iloc[0] for s in examples]
        selected += [row for _, row in views.iterrows()]
        images = []
        for i, row in enumerate(selected):
            path = folder / f"image_{i+1:02d}.png"
            prepare_crop(Path(row.path), cameras.loc[row.camera], path)
            images.append(path)
        (folder / "prompt.txt").write_text(prompt)
        record = {"sample_id": query_id, "split": "train" if query_id in set(truth.sample_id) else "test",
                  "examples": examples, "state": "requested", "folder": str(folder),
                  "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                  "image_sha256": [sha256(p.read_bytes()).hexdigest() for p in images]}
        ledger["requests"].append(record)
        write_json(ledger, ledger_path)
        try:
            with (folder / "events.jsonl").open("w") as stdout, (folder / "stderr.txt").open("w") as stderr:
                result = subprocess.run(cli_command(folder, images, schema), input=prompt, text=True,
                                        stdout=stdout, stderr=stderr, timeout=300)
            if result.returncode:
                raise RuntimeError(f"Model client returned {result.returncode}")
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
            usage = audit_events(events)
            record.update(usage)
            response = folder / "response.json"
            curve = parse_prediction(response.read_text())
            record.update(state="complete", curve=curve.tolist(), response_path=str(response),
                          response_sha256=sha256(response.read_bytes()).hexdigest())
        except Exception as error:
            record.update(state="failed", error_type=type(error).__name__, error=str(error))
            write_json(ledger, ledger_path)
            raise
        write_json(ledger, ledger_path)
        print(f"Visual-language completed {len(ledger['requests'])}/34; input tokens {usage['input_tokens']}.", flush=True)

    records = {r["sample_id"]: r for r in ledger["requests"]}
    oof = np.array([records[s]["curve"] for s in truth.sample_id])
    frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, "sample_id", truth.sample_id.tolist())
    frame["emd"] = [emd_score(y, p) for y, p in zip(curve_array(truth), oof, strict=True)]
    frame.to_csv(output / "oof.csv", index=False)
    summary = {"experiment": "visual_language", "loo_emd": emd_score(curve_array(truth), oof),
               "heldout_soils": len(truth), "producing_commit": ledger["producing_commit"],
               "requests": len(ledger["requests"]) + ledger["smoke_requests"],
               "oof": {"path": str(output / "oof.csv"), "sha256": sha256((output / "oof.csv").read_bytes()).hexdigest()},
               "sources": sources, "camera_transfer_evaluated": False}
    if include_test:
        candidate = sample.copy()
        candidate[ordered_grain_columns(sample)] = [records[s]["curve"] for s in sample.sample_id]
        validate_submission(candidate, sample)
        path = output / "visual_language.csv"
        candidate.to_csv(path, index=False)
        summary["submission"] = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    write_json(summary, output / "summary.json")
    return summary
