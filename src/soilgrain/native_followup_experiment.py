"""Two fixed follow-ups to the native-resolution winner; no automatic uploads."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.distribution_experiment import record, save_candidate
from soilgrain.distribution_models import sqrt_mass_ridge
from soilgrain.io import write_json
from soilgrain.ridge import ridge_curves
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.texture_experiment import _aligned_cache


PINNED_DISTRIBUTION = "f497da50d1d6392b454151882f88246ae0407784e55581afd7831debc424ae29"


def native_lbp_curves(train_features, train_curves, query_features) -> np.ndarray:
    """Equal CDF blend of independently standardized native23 and LBP40 ridges."""
    train, query = np.asarray(train_features, dtype=float), np.asarray(query_features, dtype=float)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != 63 or query.shape[1] != 63:
        raise ValueError("Blend requires 23 native spectral features followed by 40 LBP features")
    return (ridge_curves(train[:, :23], train_curves, query[:, :23])
            + ridge_curves(train[:, 23:], train_curves, query[:, 23:])) / 2.


def run_native_followup(config_path: str = "configs/data.yaml") -> dict:
    cfg = load_config(config_path)
    output = cfg.artifacts_dir / "experiments" / "september18_round2"
    if output.exists():
        raise FileExistsError(f"Preserve existing experiment output: {output}")
    truth, sample, photos, _components, sources = load_search_inputs(config_path)
    previous = cfg.artifacts_dir / "experiments" / "distribution_search"
    summary_path = previous / "summary.json"
    sources += _verified_sources([{"path": str(summary_path), "sha256": PINNED_DISTRIBUTION}])
    previous_summary = json.loads(summary_path.read_text())
    incumbent = next(c for c in previous_summary["candidates"]
                     if c["experiment"] == "native_spectral_ridge")
    reference = pd.read_csv(incumbent["oof"]["path"])
    verified = {Path(item["path"]).resolve() for item in sources}
    frames = []
    for name, count in (("native_spectral", 23), ("lbp", 40)):
        path = previous / f"{name}_photo_features.csv"
        if path.resolve() not in verified:
            raise ValueError(f"Feature cache is not a verified source: {path}")
        frames.append(_aligned_cache(path, photos["spectral"], count))
    native, lbp = frames
    combined = pd.concat([
        native, lbp[[f"feature_{i}" for i in range(40)]].rename(
            columns={f"feature_{i}": f"feature_{i + 23}" for i in range(40)}),
    ], axis=1)
    output.mkdir(parents=True)
    candidates = [save_candidate(name, frame, predictor, truth, sample, reference, output)
                  for name, frame, predictor in (
        ("native_sqrt_mass_ridge", native, sqrt_mass_ridge),
        ("native_lbp_blend", combined, native_lbp_curves),
    )]
    sources.append(record(Path(__file__)))
    summary = {
        "producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "declaration": record(Path("docs/september18-round2-plan.md")),
        "sources": sources, "candidates": candidates,
        "train_soils": len(truth), "test_soils": len(sample),
        "reference": "native_spectral_ridge",
        "selection": "Fixed recipes; no parameter or blend-weight search; no automatic upload",
    }
    write_json(summary, output / "summary.json")
    return summary


if __name__ == "__main__":
    run_native_followup()
