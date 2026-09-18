"""The declared native-feature exemplar assay; 34 isolated requests, no uploads."""
from __future__ import annotations

import fcntl
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns
from soilgrain.texture_experiment import _aligned_cache
from soilgrain.visual_language import (
    SCHEMA, audit_cached_record, audit_events, build_prompt, cli_command, parse_prediction,
    prepare_crop,
)


PINNED_ROUND2 = "a9e7fd09e80fb54e8048368a0981af3283f4a33f06d0573aede0225308c32554"
PINNED_ORIGINAL = "dd4d650236ea87303248170937587129de988ef7f181e063af9abbaa1d6b86b3"


def file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def save_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Preserve existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def json_bytes(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()


def save_ledger(ledger: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(json_bytes(ledger))
    temporary.replace(path)


def load_inputs(config_path: str):
    cfg = load_config(config_path)
    truth, sample, photos, _components, sources = load_search_inputs(config_path)
    sources += _verified_sources([
        {"path": str(cfg.artifacts_dir / "experiments/september18_round2/summary.json"),
         "sha256": PINNED_ROUND2},
        {"path": str(cfg.artifacts_dir / "experiments/visual_language/requests.json"),
         "sha256": PINNED_ORIGINAL},
    ])
    cache = cfg.artifacts_dir / "experiments/distribution_search/native_spectral_photo_features.csv"
    if cache.resolve() not in {Path(s["path"]).resolve() for s in sources}:
        raise ValueError("Native feature cache must have pinned provenance")
    native = _aligned_cache(cache, photos["spectral"], 23)
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    sources.append(file_record(Path("docs/september18-round3-plan.md")))
    return cfg, truth, sample, native, cameras, sources


def runtime_identity() -> dict:
    paths = [Path(__file__).with_name(name) for name in (
        "retrieval_visual_language.py", "visual_language.py", "config.py", "constants.py",
        "metrics.py", "search_inputs.py", "submission.py", "targets.py", "texture_experiment.py",
    )]
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    code = [file_record(path.resolve()) for path in paths]
    for path, item in zip(paths, code, strict=True):
        committed = subprocess.check_output(["git", "show", f"HEAD:{path.resolve().relative_to(root)}"])
        if sha256(committed).hexdigest() != item["sha256"]:
            raise ValueError("Commit the producing code before any model request")
    return {"client_version": subprocess.check_output(["/opt/homebrew/bin/codex", "--version"], text=True).strip(),
            "producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "code_sources": code}


def select_examples(photos: pd.DataFrame, train_ids: list[str], query_id: str) -> dict:
    columns = [f"feature_{i}" for i in range(23)]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Training soil IDs must be distinct")
    training = photos.loc[(photos.split == "train") & (photos.sample_id != query_id)]
    available = sorted(s for s in train_ids if s != query_id)
    if len(available) < 8 or set(training.sample_id) != set(available):
        raise ValueError("Require all training soils and at least eight exemplars")
    means = training.groupby("sample_id")[columns].mean().loc[available].to_numpy()
    query = photos.loc[photos.sample_id == query_id, columns].mean().to_numpy()
    if not np.isfinite(means).all() or not np.isfinite(query).all():
        raise ValueError("Require finite native features and query coverage")
    center, scale = means.mean(axis=0), means.std(axis=0, ddof=0)
    scale[scale == 0] = 1
    distances = np.sum(((means - query) / scale) ** 2, axis=1)
    order = np.argsort(distances, kind="stable")[:8]
    return {"examples": [available[i] for i in order], "distances": distances[order].tolist(),
            "scaler_training_ids": available, "scaler_mean": center.tolist(),
            "scaler_std": scale.tolist()}


def build_retrieval_prompt(truth: pd.DataFrame, query_id: str, query_views: int,
                           examples: list[str]) -> str:
    if (len(examples) != 8 or len(set(examples)) != 8 or query_id in examples
            or not set(examples).issubset(set(truth.sample_id))):
        raise ValueError("Require eight distinct training examples excluding the query")
    original, _ = build_prompt(truth, query_id, query_views)
    lines = original.splitlines()
    values = curve_array(truth.set_index("sample_id").loc[examples])
    lines[1:9] = [f"Example image {i} mass percentages passing: {row.tolist()}"
                  for i, row in enumerate(values, 1)]
    return "\n".join(lines)


def request_context(truth, photos, query_id):
    selection = select_examples(photos, truth.sample_id.tolist(), query_id)
    index = photos.sort_values(["camera", "path"])
    views = index.loc[index.sample_id == query_id].groupby("camera", sort=True).head(1)
    selected = [index.loc[(index.split == "train") & (index.sample_id == s)].iloc[0]
                for s in selection["examples"]]
    selected += [row for _, row in views.iterrows()]
    prompt = build_retrieval_prompt(truth, query_id, len(views), selection["examples"])
    selection["image_sources"] = [{**file_record(Path(row.path)), "camera": row.camera}
                                  for row in selected]
    return prompt, selection, selected


def render_images(selected, cameras, folder):
    images = []
    for i, row in enumerate(selected, 1):
        path = folder / f"image_{i:02d}.png"
        prepare_crop(Path(row.path), cameras.loc[row.camera], path)
        images.append(path)
    return images


def response_usage(folder: Path) -> dict:
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    usage = audit_events(events)
    messages = [e["item"]["text"] for e in events if e.get("type") == "item.completed"
                and e.get("item", {}).get("type") == "agent_message"]
    response = (folder / "response.json").read_text()
    if len(messages) != 1 or messages[0].strip() != response.strip():
        raise ValueError("Final response differs from the completed event record")
    parse_prediction(response)
    return usage


def audit_record(record, folder, prompt, selection, selected, cameras):
    if (record["folder"] != str(folder) or record["response_path"] != str(folder / "response.json")
            or any(record[key] != value for key, value in selection.items())
            or record["prompt_sha256"] != sha256(prompt.encode()).hexdigest()):
        raise ValueError("Recorded request differs from reconstructed source context")
    audit_cached_record(record)
    _verified_sources(record["raw_files"])
    usage = response_usage(folder)
    if any(record[key] != value for key, value in usage.items()):
        raise ValueError("Recorded usage differs from completed events")
    with TemporaryDirectory(prefix="soilgrain-context-") as temporary:
        images = render_images(selected, cameras, Path(temporary))
        if [file_record(p)["sha256"] for p in images] != record["image_sha256"]:
            raise ValueError("Recorded images differ from reconstructed source crops")


def token_totals(ledger):
    return {key: sum(r.get(key, 0) for r in ledger["requests"])
            for key in ("input_tokens", "output_tokens", "reasoning_output_tokens")}


def run_retrieval_visual_language(config_path: str = "configs/data.yaml") -> dict:
    cfg, truth, sample, photos, cameras, sources = load_inputs(config_path)
    _verified_sources(sources)
    train_ids, test_ids = truth.sample_id.tolist(), sample.sample_id.tolist()
    if (len(train_ids) != 24 or len(test_ids) != 10 or len(set(train_ids + test_ids)) != 34):
        raise ValueError("The declaration requires 24 distinct holdouts and ten distinct test soils")
    identity = runtime_identity()
    output = (cfg.artifacts_dir / "experiments/september18_round3/retrieval_vlm").resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(output, truth, sample, photos, cameras, sources, identity)


def _run(output, truth, sample, photos, cameras, sources, identity):
    ledger_path = output / "requests.json"
    schema = output / "schema.json"
    query_ids = truth.sample_id.tolist() + sample.sample_id.tolist()
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        for key in ("client_version", "code_sources"):
            if ledger[key] != identity[key]:
                raise ValueError(f"Recorded {key} changed; cannot mix resumed recipes")
        if ledger["sources"] != sources or ledger["query_ids"] != query_ids:
            raise ValueError("Recorded input sources changed")
        if any(r["state"] != "complete" for r in ledger["requests"]):
            raise ValueError("Unfinished/failed request: no automatic retry")
        if [r["sample_id"] for r in ledger["requests"]] != query_ids[:len(ledger["requests"])]:
            raise ValueError("Requests must be a distinct, ordered prefix of declared soils")
        if len(ledger["requests"]) > 34:
            raise ValueError("Declared request budget exceeded")
    else:
        if any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Existing output without a ledger; no automatic retry")
        ledger = {**identity, "sources": sources, "query_ids": query_ids, "requests": [],
                  "model": "gpt-6-astra", "reasoning": "low", "max_requests": 34,
                  "max_input_tokens": 1_000_000, "max_output_tokens": 50_000}
        save_ledger(ledger, ledger_path)
    save_once(schema, json_bytes(SCHEMA))
    # Audit every completed response before allowing any fresh request.
    for number, record in enumerate(ledger["requests"]):
        context = request_context(truth, photos, query_ids[number])
        audit_record(record, output / "requests" / f"query_{number:02d}", *context, cameras)
    try:
        for number in range(len(ledger["requests"]), len(query_ids)):
            totals = token_totals(ledger)
            if (number >= 34 or totals["input_tokens"] >= 1_000_000
                    or totals["output_tokens"] >= 50_000):
                raise ValueError("Declared aggregate request/token budget reached")
            query_id = query_ids[number]
            prompt, selection, selected = request_context(truth, photos, query_id)
            folder = output / "requests" / f"query_{number:02d}"
            folder.mkdir(parents=True, exist_ok=False)
            images = render_images(selected, cameras, folder)
            save_once(folder / "prompt.txt", prompt.encode())
            record = {"sample_id": query_id, "split": "train" if number < 24 else "test",
                      **selection, "state": "requested", "folder": str(folder),
                      "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                      "image_sha256": [file_record(p)["sha256"] for p in images]}
            # Both the immutable receipt and ledger precede the external request.
            save_once(output / "receipts" / f"query_{number:02d}.json", json_bytes(record))
            ledger["requests"].append(record)
            save_ledger(ledger, ledger_path)
            try:
                with (folder / "events.jsonl").open("x") as stdout, (folder / "stderr.txt").open("x") as stderr:
                    result = subprocess.run(cli_command(folder, images, schema), input=prompt, text=True,
                                            stdout=stdout, stderr=stderr, timeout=300)
                if result.returncode:
                    raise RuntimeError(f"Model client returned {result.returncode}")
                record.update(audit_events([json.loads(line) for line in
                                            (folder / "events.jsonl").read_text().splitlines()]))
                response_usage(folder)
                response = folder / "response.json"
                record.update(state="complete", curve=parse_prediction(response.read_text()).tolist(),
                              response_path=str(response), response_sha256=file_record(response)["sha256"])
            except Exception as error:
                record.update(state="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                record["raw_files"] = [file_record(p) for p in sorted(folder.iterdir()) if p.is_file()]
                save_once(output / "records" / f"query_{number:02d}.json", json_bytes(record))
                save_ledger(ledger, ledger_path)
            print(f"Retrieval visual-language completed {number + 1}/34; {token_totals(ledger)}", flush=True)
        totals = token_totals(ledger)
        if totals["input_tokens"] > 1_000_000 or totals["output_tokens"] > 50_000:
            raise ValueError("Final response exceeded the aggregate token budget")
    except Exception as error:
        failure = {"eligible": False, "error_type": type(error).__name__, "error": str(error),
                   "requests": len(ledger["requests"]), **token_totals(ledger),
                   "fallback": "native_sqrt_mass_ridge", "ledger": file_record(ledger_path)}
        save_once(output / "failure.json", json_bytes(failure))
        raise

    oof = np.array([r["curve"] for r in ledger["requests"][:24]])
    frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, "sample_id", truth.sample_id.tolist())
    frame["emd"] = [emd_score(y, p) for y, p in zip(curve_array(truth), oof, strict=True)]
    candidate = sample.copy()
    candidate[ordered_grain_columns(sample)] = [r["curve"] for r in ledger["requests"][24:]]
    validate_submission(candidate, sample)
    save_once(output / "oof.csv", frame.to_csv(index=False).encode())
    save_once(output / "retrieval_visual_language.csv", candidate.to_csv(index=False).encode())
    summary = {"experiment": "retrieval_visual_language", "eligible": True,
               "loo_emd": emd_score(curve_array(truth), oof), "heldout_soils": 24, "test_soils": 10,
               "requests": 34, **token_totals(ledger), "producing_commit": ledger["producing_commit"],
               "oof": file_record(output / "oof.csv"),
               "submission": file_record(output / "retrieval_visual_language.csv"),
               "sources": sources + ledger["code_sources"] + [file_record(ledger_path)]
                          + [file_record(p) for p in sorted((output / "records").glob("*.json"))]
                          + [file_record(p) for p in sorted((output / "receipts").glob("*.json"))],
               "camera_transfer_evaluated": False}
    save_once(output / "summary.json", json_bytes(summary))
    return summary


if __name__ == "__main__":
    run_retrieval_visual_language()
