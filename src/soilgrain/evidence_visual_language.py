"""The two declared September 23 visual-language evidence assays."""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.context_visual_language import request_context as full_context
from soilgrain.metrics import emd_score
from soilgrain.retrieval_visual_language import (
    file_record, json_bytes, save_ledger, save_once, token_totals,
)
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns
from soilgrain.visual_language import (
    SCHEMA, audit_cached_record, audit_events, build_prompt, cli_command, parse_prediction,
    prepare_crop,
)


RECIPES = ("paired_examples_vlm", "complete_context_vlm")
MAX_REQUESTS = 34
MAX_INPUT_TOKENS = 2_000_000
MAX_OUTPUT_TOKENS = 50_000


def request_context(recipe: str, truth: pd.DataFrame, photos: pd.DataFrame, query_id: str):
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    if recipe == "complete_context_vlm":
        return full_context("all_examples_vlm", truth, photos, query_id)
    index = photos.sort_values(["camera", "path"])
    views = index.loc[index.sample_id == query_id].groupby("camera", sort=True).head(1)
    if views.empty:
        raise ValueError("Query requires at least one camera view")
    original, examples = build_prompt(truth, query_id, len(views))
    example_rows = [row for sample_id in examples for _, row in
                    index.loc[(index.split == "train") & (index.sample_id == sample_id)]
                    .groupby("camera", sort=True).head(1).iterrows()]
    rows = example_rows + [row for _, row in views.iterrows()]
    ranges = []
    start = 1
    for sample_id in examples:
        count = sum(row.sample_id == sample_id for row in example_rows)
        if not count:
            raise ValueError("Every example requires a camera view")
        ranges.append(f"Example soil {len(ranges)+1} (images {start}-{start+count-1}) mass percentages passing: ")
        start += count
    values = curve_array(truth.set_index("sample_id").loc[examples])
    lines = original.splitlines()
    lines[0] = lines[0].replace(
        "The first 8 images are independent labeled examples, one photo per soil. ",
        "The first images are eight independent labeled soils; each soil's camera views are consecutive "
        "and share one laboratory CDF. ")
    lines[1:9] = [f"{label}{value.tolist()}" for label, value in zip(ranges, values, strict=True)]
    lines.insert(9, f"Query images are {start}-{start+len(views)-1}, one per camera.")
    prompt = "\n".join(lines)
    selection = {
        "examples": examples,
        "image_sources": [{**file_record(Path(row.path)), "camera": row.camera,
                           "widths_mm": [100]} for row in rows],
    }
    return prompt, selection, [(row, 100) for row in rows]


def render_images(selected, cameras: pd.DataFrame, folder: Path) -> list[Path]:
    images = []
    for number, (row, _width_mm) in enumerate(selected, 1):
        output = folder / f"image_{number:02d}.png"
        prepare_crop(Path(row.path), cameras.loc[row.camera], output)
        images.append(output)
    return images


RECONNECT = re.compile(
    r"Reconnecting\.\.\. [1-5]/5 \(stream disconnected before completion: "
    r"websocket closed by server before response\.completed\)"
)
FALLBACK = ("Falling back from WebSockets to HTTPS transport. stream disconnected "
            "before completion: websocket closed by server before response.completed")
CODE_MODE = ("Code Mode is unavailable because code-mode host is disabled. Code mode "
             "will fail closed; enable `features.code_mode_host` and install "
             "`codex-code-mode-host`.")
STDERR_RETRY = re.compile(
    r"codex_core::responses_retry: stream disconnected - retrying sampling request "
    r"\(([1-5])/5 in \d+(?:\.\d+)?(?:ms|s)\)\.\.\. turn_id=[0-9a-f-]+ "
    r"retries=([1-5]) max_retries=5 sampling_error=stream disconnected before completion: "
    r"(?:websocket closed by server before response\.completed|"
    r"WebSocket protocol error: Connection reset without closing handshake)"
)


def stderr_transport_usage(path: Path) -> dict:
    counts = {"stderr_retry_warnings": 0, "stderr_fallback_warnings": 0}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"\S+\s+(WARN|ERROR)\s+(.+)", line)
        if not match or match[1] != "WARN":
            raise ValueError("Unexpected CLI stderr severity or line")
        message = match[2]
        retry = STDERR_RETRY.fullmatch(message)
        if retry and retry[1] == retry[2]:
            counts["stderr_retry_warnings"] += 1
        elif message == "codex_core::client: falling back to HTTP":
            counts["stderr_fallback_warnings"] += 1
        elif (message == "codex_rollout::list: state db discrepancy during "
              "find_thread_path_by_id_str_in_subdir: falling_back"
              or message.startswith("codex_core::shell_snapshot: Failed to delete shell snapshot at ")):
            continue
        else:
            raise ValueError("Unexpected CLI stderr warning")
    return counts


def transport_events(events: list[dict]) -> tuple[list[dict], int]:
    filtered, transport = [], 0
    for event in events:
        kind, item = event.get("type"), event.get("item", {})
        if kind == "error" and RECONNECT.fullmatch(event.get("message", "")):
            transport += 1
            continue
        if (kind == "item.completed" and item.get("type") == "error"
                and item.get("message") == FALLBACK):
            transport += 1
            continue
        if item.get("type") == "error" and (kind != "item.completed" or item.get("message") != CODE_MODE):
            raise ValueError("Unexpected error item")
        filtered.append(event)
    return filtered, transport


def response_usage(folder: Path) -> dict:
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    filtered, transport = transport_events(events)
    usage = audit_events(filtered)
    messages = [event["item"]["text"] for event in filtered
                if event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "agent_message"]
    response = (folder / "response.json").read_text()
    if len(messages) != 1 or messages[0].strip() != response.strip():
        raise ValueError("Require exactly one final message matching the response file")
    parse_prediction(response)
    return {**usage, "transport_events": transport,
            **stderr_transport_usage(folder / "stderr.txt")}


def audit_record(record: dict, folder: Path, prompt: str, selection: dict,
                 selected, cameras: pd.DataFrame) -> None:
    expected = {"folder": str(folder), "response_path": str(folder / "response.json"),
                "prompt_sha256": sha256(prompt.encode()).hexdigest()}
    if any(record.get(key) != value for key, value in {**selection, **expected}.items()):
        raise ValueError("Recorded request differs from reconstructed source context")
    number = folder.name
    output = folder.parents[1]
    receipt = json.loads((output / "receipts" / f"{number}.json").read_text())
    if (receipt.get("state") != "requested"
            or any(record.get(key) != value for key, value in receipt.items() if key != "state")):
        raise ValueError("Immutable request receipt differs from completed record")
    if json.loads((output / "records" / f"{number}.json").read_text()) != record:
        raise ValueError("Immutable completion record differs from ledger")
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
    plan = Path("docs/september23-plan.md")
    sources.append(file_record(plan))
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    return cfg, truth, sample, photos["spectral"], cameras, sources


def runtime_identity() -> dict:
    paths = [Path(__file__).with_name(name) for name in (
        "evidence_visual_language.py", "context_visual_language.py",
        "retrieval_visual_language.py", "visual_language.py",
        "config.py", "constants.py", "metrics.py", "search_inputs.py", "submission.py",
        "targets.py",
    )] + [Path("docs/september23-plan.md")]
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


def run_evidence_visual_language(recipe: str, config_path: str = "configs/data.yaml") -> dict:
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    cfg, truth, sample, photos, cameras, sources = load_inputs(config_path)
    _verified_sources(sources)
    query_ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    if len(query_ids) != 34 or len(truth) != 24 or len(set(query_ids)) != 34:
        raise ValueError("The declaration requires 24 distinct holdouts and ten distinct test soils")
    identity = runtime_identity()
    output = (cfg.artifacts_dir / "experiments" / "september23" / recipe).resolve()
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
    fresh_phase = False
    between_requests = True
    try:
        if (output / "failure.json").exists():
            raise ValueError("Recorded failure: no automatic retry")
        if ledger.get("recipe") != recipe or ledger.get("sources") != sources:
            raise ValueError("Recorded recipe or input sources changed")
        if any(ledger.get(key) != value for key, value in {
                "model": "gpt-6-astra", "reasoning": "low", "max_requests": MAX_REQUESTS,
                "max_input_tokens": MAX_INPUT_TOKENS,
                "max_output_tokens": MAX_OUTPUT_TOKENS}.items()):
            raise ValueError("Recorded model or budget changed")
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
        save_once(schema_path, json_bytes(SCHEMA))

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
                    try:
                        result = request(cli_command(folder, images, schema_path), input=prompt,
                                         text=True, stdout=stdout, stderr=stderr, timeout=300)
                    finally:
                        stdout.flush()
                        stderr.flush()
                        events = []
                        for line in (folder / "events.jsonl").read_text().splitlines():
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass  # Preserve the raw log; a malformed stream still fails below.
                        record["transport_events"] = sum(
                            event.get("type") == "error" and bool(RECONNECT.fullmatch(event.get("message", "")))
                            or event.get("type") == "item.completed" and event.get("item", {}).get("message") == FALLBACK
                            for event in events)
                        completed = [event for event in events if event.get("type") == "turn.completed"]
                        if len(completed) == 1 and "usage" in completed[0]:
                            usage = completed[0]["usage"]
                            record.update({key: int(usage[key]) for key in
                                           ("input_tokens", "output_tokens", "reasoning_output_tokens")})
                        record.update(stderr_transport_usage(folder / "stderr.txt"))
                if result.returncode:
                    raise RuntimeError(f"Model client returned {result.returncode}")
                record.update(response_usage(folder))
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
                   **token_totals(ledger),
                   "transport_events": sum(record.get("transport_events", 0)
                                           for record in ledger["requests"]),
                   **{key: sum(record.get(key, 0) for record in ledger["requests"])
                      for key in ("stderr_retry_warnings", "stderr_fallback_warnings")},
                   "ledger": file_record(ledger_path)}
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
               "transport_events": sum(record["transport_events"] for record in ledger["requests"]),
               **{key: sum(record[key] for record in ledger["requests"])
                  for key in ("stderr_retry_warnings", "stderr_fallback_warnings")},
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
    run_evidence_visual_language(arguments.recipe, arguments.config_path)


if __name__ == "__main__":
    main()
