"""The two declared September 22 visual-language context assays."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.metrics import emd_score
from soilgrain.retrieval_visual_language import (
    file_record, json_bytes, response_usage, save_ledger, save_once, token_totals,
)
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns
from soilgrain.visual_language import (
    SCHEMA, audit_cached_record, audit_events, build_prompt, cli_command, parse_prediction,
)


RECIPES = ("multiscale_vlm", "all_examples_vlm")
MAX_REQUESTS = 34
MAX_INPUT_TOKENS = 2_000_000
MAX_OUTPUT_TOKENS = 50_000


def prepare_physical_crop(path: Path, camera: pd.Series, output: Path, width_mm: int) -> None:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    ppm = float(camera["ppm"]) * max(image.size) / max(camera["width"], camera["height"])
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("Invalid physical calibration")
    side = round(width_mm * ppm)
    if not 1 <= side <= min(image.size):
        raise ValueError(f"{width_mm}mm crop does not fit")
    left, top = (image.width - side) // 2, (image.height - side) // 2
    crop = image.crop((left, top, left + side, top + side))
    if side > 768:
        crop = crop.resize((768, 768), Image.Resampling.LANCZOS)
    crop.info.clear()
    crop.save(output, format="PNG")


def _prompt(truth: pd.DataFrame, query_id: str, query_views: int,
            examples: list[str], recipe: str) -> str:
    if query_views < 1 or query_id in examples or len(set(examples)) != len(examples):
        raise ValueError("Require distinct examples excluding the query and at least one query view")
    values = curve_array(truth.set_index("sample_id").loc[examples])
    if recipe == "multiscale_vlm":
        layout = (
            "Each source photograph is represented by one consecutive image pair: its original "
            "calibrated 100 mm square center crop followed by its original calibrated 25 mm square "
            "center crop. The first 8 consecutive image pairs are independent labeled examples, "
            "one photo per soil. "
            f"The remaining {query_views} consecutive image pairs are query photos of the SAME new "
            "soil from different views. "
        )
        labels = [f"Example photo {i} (images {2*i-1}-{2*i}, 100 mm then 25 mm) "
                  f"mass percentages passing: {row.tolist()}"
                  for i, row in enumerate(values, 1)]
        query = [f"Query photo {i} is images {2*len(examples)+2*i-1}-"
                 f"{2*len(examples)+2*i} (100 mm then 25 mm)."
                 for i in range(1, query_views + 1)]
    elif recipe == "all_examples_vlm":
        layout = (
            "Each attached image is an original calibrated 100 mm square center crop. "
            f"The first {len(examples)} images are independent labeled examples, one photo per soil. "
            f"The remaining {query_views} query images show the SAME new soil from different views. "
        )
        labels = [f"Example image {i} mass percentages passing: {row.tolist()}"
                  for i, row in enumerate(values, 1)]
        query = []
    else:
        raise ValueError(f"Unknown recipe: {recipe}")
    introduction = (
        "Estimate a soil sample's laboratory mass-based cumulative grain size distribution from "
        "calibrated photographs. " + layout +
        "Photographs can differ in camera, resolution, color rendering and arrangement. "
        "Use the examples and visual soil texture/type to infer its distribution, including "
        "unresolved fine material. Percentages describe dry bulk MASS passing each diameter, "
        "not visible particle counts or surface area. Never identify the soil or retrieve data. "
        "Use only attached images and these examples; do not use tools or access files. "
        f"Diameters in mm, in order: {list(SUPPORT_DIAMETERS)}."
    )
    ending = (
        'Return only {"cdf": [eleven numeric percentages]} for the query soil. '
        "Values must be finite, between 0 and 100, nondecreasing, with the last exactly 100. "
        "Give one best estimate, without commentary."
    )
    return "\n".join([introduction, *labels, *query, ending])


def request_context(recipe: str, truth: pd.DataFrame, photos: pd.DataFrame, query_id: str):
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    train_ids = truth.sample_id.tolist()
    available = sorted(sample_id for sample_id in train_ids if sample_id != query_id)
    if recipe == "multiscale_vlm":
        _, examples = build_prompt(truth, query_id, 1)
        widths = (100, 25)
    else:
        examples = available
        widths = (100,)
    index = photos.sort_values(["camera", "path"])
    views = index.loc[index.sample_id == query_id].groupby("camera", sort=True).head(1)
    if views.empty:
        raise ValueError("Query requires at least one camera view")
    example_rows = [index.loc[(index.split == "train") & (index.sample_id == sample_id)].iloc[0]
                    for sample_id in examples]
    rows = example_rows + [row for _, row in views.iterrows()]
    selected = [(row, width) for row in rows for width in widths]
    prompt = _prompt(truth, query_id, len(views), examples, recipe)
    selection = {
        "examples": examples,
        "image_sources": [{**file_record(Path(row.path)), "camera": row.camera,
                           "widths_mm": list(widths)} for row in rows],
    }
    return prompt, selection, selected


def render_images(selected, cameras: pd.DataFrame, folder: Path) -> list[Path]:
    images = []
    for number, (row, width_mm) in enumerate(selected, 1):
        output = folder / f"image_{number:02d}.png"
        prepare_physical_crop(Path(row.path), cameras.loc[row.camera], output, width_mm)
        images.append(output)
    return images


def audit_record(record: dict, folder: Path, prompt: str, selection: dict,
                 selected, cameras: pd.DataFrame) -> None:
    expected = {"folder": str(folder), "response_path": str(folder / "response.json"),
                "prompt_sha256": sha256(prompt.encode()).hexdigest()}
    if any(record.get(key) != value for key, value in {**selection, **expected}.items()):
        raise ValueError("Recorded request differs from reconstructed source context")
    audit_cached_record(record)
    _verified_sources(record["raw_files"])
    usage = response_usage(folder)
    if any(record.get(key) != value for key, value in usage.items()):
        raise ValueError("Recorded usage differs from completed events")
    with TemporaryDirectory(prefix="soilgrain-context-") as temporary:
        images = render_images(selected, cameras, Path(temporary))
        if [file_record(path)["sha256"] for path in images] != record["image_sha256"]:
            raise ValueError("Recorded images differ from reconstructed source crops")


def load_inputs(config_path: str):
    cfg = load_config(config_path)
    truth, sample, photos, _components, sources = load_search_inputs(config_path)
    plan = Path("docs/september22-plan.md")
    sources.append(file_record(plan))
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    return cfg, truth, sample, photos["spectral"], cameras, sources


def runtime_identity() -> dict:
    paths = [Path(__file__).with_name(name) for name in (
        "context_visual_language.py", "retrieval_visual_language.py", "visual_language.py",
        "config.py", "constants.py", "metrics.py", "search_inputs.py", "submission.py",
        "targets.py",
    )] + [Path("docs/september22-plan.md")]
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    code_sources = [file_record(path.resolve()) for path in paths]
    for path, record in zip(paths, code_sources, strict=True):
        committed = subprocess.check_output(["git", "show", f"HEAD:{path.resolve().relative_to(root)}"])
        if sha256(committed).hexdigest() != record["sha256"]:
            raise ValueError("Commit the producing code and declaration before any model request")
    client = Path("/opt/homebrew/bin/codex").resolve()
    version = subprocess.check_output([str(client), "--version"], text=True).strip()
    if version != "codex-cli 0.154.0":
        raise ValueError("Codex client version differs from the declaration")
    return {"client_version": version, "client_binary": file_record(client),
            "producing_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "code_sources": code_sources}


def run_context_visual_language(recipe: str, config_path: str = "configs/data.yaml") -> dict:
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    cfg, truth, sample, photos, cameras, sources = load_inputs(config_path)
    _verified_sources(sources)
    query_ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    if len(query_ids) != 34 or len(truth) != 24 or len(set(query_ids)) != 34:
        raise ValueError("The declaration requires 24 distinct holdouts and ten distinct test soils")
    identity = runtime_identity()
    output = (cfg.artifacts_dir / "experiments" / "september22" / recipe).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(output, recipe, truth, sample, photos, cameras, sources, identity,
                    subprocess.run)


def _run(output: Path, recipe: str, truth: pd.DataFrame, sample: pd.DataFrame,
         photos: pd.DataFrame, cameras: pd.DataFrame, sources: list[dict], identity: dict,
         request, progress=print) -> dict:
    ledger_path = output / "requests.json"
    schema_path = output / "schema.json"
    query_ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    else:
        if any(path.name != ".lock" for path in output.iterdir()):
            raise ValueError("Existing output without a ledger; no automatic retry")
        ledger = {**identity, "recipe": recipe, "sources": sources, "query_ids": query_ids,
                  "requests": [], "model": "gpt-6-astra", "reasoning": "low",
                  "max_requests": MAX_REQUESTS, "max_input_tokens": MAX_INPUT_TOKENS,
                  "max_output_tokens": MAX_OUTPUT_TOKENS}
        save_ledger(ledger, ledger_path)
    save_once(schema_path, json_bytes(SCHEMA))

    fresh_phase = False
    between_requests = True
    try:
        if (output / "failure.json").exists():
            raise ValueError("Recorded failure: no automatic retry")
        if ledger.get("recipe") != recipe or ledger.get("sources") != sources:
            raise ValueError("Recorded recipe or input sources changed")
        if ledger.get("query_ids") != query_ids:
            raise ValueError("Recorded query order changed")
        for key, value in identity.items():
            if ledger.get(key) != value:
                raise ValueError(f"Recorded {key} changed; cannot resume")
        if any(record.get("state") != "complete" for record in ledger["requests"]):
            raise ValueError("Unfinished/failed request: no automatic retry")
        if [record.get("sample_id") for record in ledger["requests"]] != query_ids[:len(ledger["requests"])]:
            raise ValueError("Requests must be the verified ordered prefix")
        if len(ledger["requests"]) > MAX_REQUESTS:
            raise ValueError("Declared request budget exceeded")

        for number, record in enumerate(ledger["requests"]):
            context = request_context(recipe, truth, photos, query_ids[number])
            audit_record(record, output / "requests" / f"query_{number:02d}", *context, cameras)

        for number in range(len(ledger["requests"]), len(query_ids)):
            fresh_phase = True
            between_requests = False
            totals = token_totals(ledger)
            if (number >= MAX_REQUESTS or totals["input_tokens"] >= MAX_INPUT_TOKENS
                    or totals["output_tokens"] >= MAX_OUTPUT_TOKENS):
                raise ValueError("Declared aggregate request/token budget reached")
            query_id = query_ids[number]
            prompt, selection, selected = request_context(recipe, truth, photos, query_id)
            folder = output / "requests" / f"query_{number:02d}"
            folder.mkdir(parents=True, exist_ok=False)
            images = render_images(selected, cameras, folder)
            save_once(folder / "prompt.txt", prompt.encode())
            record = {"sample_id": query_id, "split": "train" if number < 24 else "test",
                      **selection, "state": "requested", "folder": str(folder),
                      "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                      "image_sha256": [file_record(path)["sha256"] for path in images]}
            save_once(output / "receipts" / f"query_{number:02d}.json", json_bytes(record))
            ledger["requests"].append(record)
            save_ledger(ledger, ledger_path)
            try:
                with (folder / "events.jsonl").open("x") as stdout, \
                        (folder / "stderr.txt").open("x") as stderr:
                    result = request(cli_command(folder, images, schema_path), input=prompt, text=True,
                                     stdout=stdout, stderr=stderr, timeout=300)
                if result.returncode:
                    raise RuntimeError(f"Model client returned {result.returncode}")
                events = [json.loads(line) for line in
                          (folder / "events.jsonl").read_text().splitlines()]
                record.update(audit_events(events))
                response_usage(folder)
                response = folder / "response.json"
                record.update(state="complete", curve=parse_prediction(response.read_text()).tolist(),
                              response_path=str(response),
                              response_sha256=file_record(response)["sha256"])
            except BaseException as error:
                record.update(state="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                record["raw_files"] = [file_record(path) for path in sorted(folder.iterdir())
                                       if path.is_file()]
                save_once(output / "records" / f"query_{number:02d}.json", json_bytes(record))
                save_ledger(ledger, ledger_path)
            between_requests = True
            progress(f"{recipe} completed {number + 1}/34; {token_totals(ledger)}", flush=True)

        totals = token_totals(ledger)
        if totals["input_tokens"] > MAX_INPUT_TOKENS or totals["output_tokens"] > MAX_OUTPUT_TOKENS:
            raise ValueError("Final response exceeded the aggregate token budget")
    except BaseException as error:
        recorded_incomplete = any(record.get("state") != "complete"
                                  for record in ledger["requests"])
        if ((not fresh_phase and not recorded_incomplete)
                or (isinstance(error, KeyboardInterrupt) and between_requests
                    and all(record.get("state") == "complete" for record in ledger["requests"]))):
            raise
        failure = {"eligible": False, "experiment": recipe, "error_type": type(error).__name__,
                   "error": str(error), "requests": len(ledger["requests"]),
                   **token_totals(ledger), "ledger": file_record(ledger_path)}
        failure_path = output / "failure.json"
        if not failure_path.exists():
            save_once(failure_path, json_bytes(failure))
        raise

    oof = np.asarray([record["curve"] for record in ledger["requests"][:24]])
    frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, "sample_id", truth.sample_id.tolist())
    frame["emd"] = [emd_score(actual, predicted)
                    for actual, predicted in zip(curve_array(truth), oof, strict=True)]
    candidate = sample.copy()
    candidate[ordered_grain_columns(sample)] = [record["curve"]
                                                 for record in ledger["requests"][24:]]
    validate_submission(candidate, sample)
    oof_path, submission_path = output / "oof.csv", output / f"{recipe}.csv"
    save_once(oof_path, frame.to_csv(index=False).encode())
    save_once(submission_path, candidate.to_csv(index=False).encode())
    summary = {"experiment": recipe, "eligible": True,
               "loo_emd": emd_score(curve_array(truth), oof), "heldout_soils": 24,
               "test_soils": 10, "requests": 34, **token_totals(ledger),
               "producing_commit": ledger["producing_commit"], "oof": file_record(oof_path),
               "submission": file_record(submission_path),
               "sources": sources + ledger["code_sources"] + [file_record(ledger_path)]
                          + [file_record(path) for path in sorted((output / "records").glob("*.json"))]
                          + [file_record(path) for path in sorted((output / "receipts").glob("*.json"))],
               "camera_transfer_evaluated": False}
    save_once(output / "summary.json", json_bytes(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--config-path", default="configs/data.yaml")
    arguments = parser.parse_args()
    run_context_visual_language(arguments.recipe, arguments.config_path)


if __name__ == "__main__":
    main()
