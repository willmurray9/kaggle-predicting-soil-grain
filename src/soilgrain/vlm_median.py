"""The fixed September 23 median of three saved, whole-soil VLM predictions."""
from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.retrieval_visual_language import file_record, json_bytes, save_once
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


SOURCES = {
    "artifacts/experiments/visual_language/summary.json":
        "e296c4503992a96efd3b8c3b5cde70c6ded08726c7c3b61701059c461d441369",
    "artifacts/experiments/september18_round3/retrieval_vlm/summary.json":
        "370d990ae99b6a7d68a1402a3adccb6422169115d7c150b466cd61bf90c7d542",
    "artifacts/experiments/september22/multiscale_vlm/summary.json":
        "7b3f4f8d4cc6817a45046faf2c58a23275ee3e3893114d5e470fd63c2d439b9b",
}


def aligned_median(frames: list[pd.DataFrame], ids: list[str]) -> pd.DataFrame:
    if len(frames) != 3 or not ids or len(set(ids)) != len(ids):
        raise ValueError("Require exactly three components and distinct expected soil IDs")
    values = []
    for frame in frames:
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(ids):
            raise ValueError("Component soil IDs differ from the expected coverage")
        validate_cumulative_curves(frame)
        values.append(curve_array(frame.set_index("sample_id").loc[ids]))
    result = pd.DataFrame(np.median(values, axis=0), columns=CANONICAL_GRAIN_LABELS)
    result.insert(0, "sample_id", ids)
    validate_cumulative_curves(result)
    return result


def run_median() -> dict:
    output = Path("artifacts/experiments/september23/vlm_median")
    if output.exists():
        raise FileExistsError(f"Preserve existing output: {output}")
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    code = [file_record(path) for path in sorted(Path("src/soilgrain").glob("*.py"))]
    code.append(file_record(Path("docs/september23-plan.md")))
    for record in code:
        relative = Path(record["path"]).resolve().relative_to(root)
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"])
        if sha256(committed).hexdigest() != record["sha256"]:
            raise ValueError("Commit producing code and declaration before generating predictions")
    truth, sample, _, _, inputs = load_search_inputs()
    sources = _verified_sources([{"path": path, "sha256": digest}
                                 for path, digest in SOURCES.items()])
    summaries = [json.loads(Path(path).read_text()) for path in SOURCES]
    oof = aligned_median([pd.read_csv(s["oof"]["path"]) for s in summaries], truth.sample_id.tolist())
    test = aligned_median([pd.read_csv(s["submission"]["path"]) for s in summaries],
                          sample.sample_id.tolist())
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = curve_array(test)
    validate_submission(submission, sample)
    oof["emd"] = [emd_score(y, p) for y, p in zip(curve_array(truth), curve_array(oof), strict=True)]
    oof_path, csv_path = output / "oof.csv", output / "submission.csv"
    save_once(oof_path, oof.to_csv(index=False).encode())
    save_once(csv_path, submission.to_csv(index=False).encode())
    summary = {"experiment": "vlm_median", "eligible": True, "heldout_soils": len(truth),
               "test_soils": len(sample), "loo_emd": float(oof.emd.mean()),
               "producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "oof": file_record(oof_path), "submission": file_record(csv_path),
               "sources": sources + inputs + code, "camera_transfer_evaluated": False,
               "recipe": "Coordinatewise median of original, retrieval and multiscale VLM CDFs; no fit"}
    save_once(output / "summary.json", json_bytes(summary))
    return summary


if __name__ == "__main__":
    result = run_median()
    print(json.dumps({key: result[key] for key in ("experiment", "loo_emd", "submission")}))
