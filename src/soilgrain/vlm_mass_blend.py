"""Fixed equal blend of the saved VLM and native square-root mass predictions."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.distribution_experiment import record
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.model_blend import _blend_predictions
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns
from soilgrain.visual_language import audit_cached_record


def blend_frames(truth, sample, vlm, mass) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = _blend_predictions(vlm["oof"], mass["oof"], ["sample_id"])
    if set(oof.sample_id) != set(truth.sample_id):
        raise ValueError("Held-out blend coverage must match every training soil")
    oof = oof.set_index("sample_id").loc[truth.sample_id].reset_index()
    actual = curve_array(truth)
    for column, frame in (("emd", oof), ("vlm_emd", vlm["oof"]), ("mass_emd", mass["oof"])):
        predictions = curve_array(frame.set_index("sample_id").loc[truth.sample_id])
        oof[column] = [emd_score(y, p) for y, p in zip(actual, predictions, strict=True)]
    for component in (vlm, mass):
        validate_submission(component["submission"], sample)
    blended = _blend_predictions(vlm["submission"], mass["submission"], ["sample_id"])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = curve_array(
        blended.set_index("sample_id").loc[sample.sample_id])
    validate_submission(submission, sample)
    return oof, submission


def run_vlm_mass_blend(config_path: str = "configs/data.yaml") -> dict:
    cfg = load_config(config_path)
    output = cfg.artifacts_dir / "experiments" / "september18_round3" / "vlm_mass_blend"
    if output.exists():
        raise FileExistsError(f"Preserve existing experiment output: {output}")
    truth, sample, _photos, _components, sources = load_search_inputs(config_path)
    paths = [cfg.artifacts_dir / "experiments" / name for name in (
        "visual_language/summary.json", "september18_round2/summary.json", "visual_language/requests.json")]
    pins = ("e296c4503992a96efd3b8c3b5cde70c6ded08726c7c3b61701059c461d441369",
            "a9e7fd09e80fb54e8048368a0981af3283f4a33f06d0573aede0225308c32554",
            "dd4d650236ea87303248170937587129de988ef7f181e063af9abbaa1d6b86b3")
    sources += _verified_sources([{"path": str(p), "sha256": digest}
                                  for p, digest in zip(paths, pins, strict=True)])
    vlm_summary, numerical, ledger = [json.loads(p.read_text()) for p in paths]
    for request in ledger["requests"]:
        if request["state"] != "complete":
            raise ValueError("VLM source has an incomplete request")
        audit_cached_record(request)
    mass_summary = next(c for c in numerical["candidates"] if c["experiment"] == "native_sqrt_mass_ridge")
    components = [{kind: pd.read_csv(summary[kind]["path"]) for kind in ("oof", "submission")}
                  for summary in (vlm_summary, mass_summary)]
    oof, submission = blend_frames(truth, sample, *components)
    files = {kind: record(write_csv(frame, output / f"{kind}.csv"))
             for kind, frame in (("oof", oof), ("submission", submission))}
    sources += [record(Path(__file__)), record(Path(__file__).with_name("model_blend.py")),
                record(Path(__file__).with_name("visual_language.py"))]
    improvements = {}
    for name, column in (("visual_language", "vlm_emd"), ("native_sqrt_mass_ridge", "mass_emd")):
        gains = (oof[column] - oof.emd).to_numpy()
        improvements[name] = {"mean_emd": float(gains.mean()), "median_emd": float(np.median(gains)),
                              "soils_improved": int((gains > 0).sum()),
                              "mean_excluding_largest": float(np.delete(gains, np.argmax(gains)).mean())}
    summary = {"experiment": "vlm_mass_blend", "weights": [0.5, 0.5],
               "loo_emd": float(oof.emd.mean()), "improvements": improvements,
               "heldout_soils": len(truth), "test_soils": len(sample), "camera_transfer_evaluated": False,
               "producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "declaration": record(Path("docs/september18-round3-plan.md")), "sources": sources, **files}
    write_json(summary, output / "summary.json")
    print(f"Fixed VLM/mass blend: {summary['loo_emd']:.5f} EMD", flush=True)
    return summary


if __name__ == "__main__":
    run_vlm_mass_blend()
