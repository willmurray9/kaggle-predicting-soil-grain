from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def _blend_predictions(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average the two curves after exact soil or soil/camera alignment."""
    for frame in (left, right):
        if frame.empty or any(key not in frame for key in keys):
            raise ValueError("Blend components must be nonempty and contain alignment keys")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Blend component keys must be nonnull and unique")
        validate_cumulative_curves(frame)
    left_indexed, right_indexed = left.set_index(keys), right.set_index(keys)
    if set(left_indexed.index) != set(right_indexed.index):
        raise ValueError("Blend component key coverage differs")
    output = left[keys].reset_index(drop=True).copy()
    output[list(CANONICAL_GRAIN_LABELS)] = (
        curve_array(left) + curve_array(right_indexed.loc[left_indexed.index])
    ) / 2.0
    validate_cumulative_curves(output)
    return output


def write_model_blend(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    """Evaluate the declared 50/50 multicrop and nested-ridge prediction blend."""
    cfg = load_config(config_path)
    train_path, sample_path = [cfg.curated_file(key) for key in ("train", "sample_submission")]
    train, sample = [pd.read_csv(path) for path in (train_path, sample_path)]
    validate_cumulative_curves(train)
    for frame in (train, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels and template must have nonempty, unique, nonnull soil IDs")
    train_ids = train["sample_id"].tolist()
    truth = train.set_index("sample_id")
    sources = [Path(config_path), train_path, sample_path]
    components = {}
    for name, filename in (
        ("multicrop", "multicrop_baseline.csv"),
        ("nested_ridge", "frozen_resnet18_nested_ridge.csv"),
    ):
        directory = cfg.artifacts_dir / "experiments" / name
        paths = [directory / "oof_predictions.csv", directory / "camera_predictions.csv",
                 directory / "summary.json", cfg.submissions_dir / filename]
        sources.extend(paths)
        components[name] = [pd.read_csv(path) for path in (paths[0], paths[1], paths[3])]
        validate_submission(components[name][2], sample)

    oof = _blend_predictions(components["multicrop"][0], components["nested_ridge"][0], ["sample_id"])
    cameras = _blend_predictions(
        components["multicrop"][1], components["nested_ridge"][1], ["sample_id", "camera"],
    )
    for frame in (oof, cameras):
        if set(frame["sample_id"]) != set(train_ids):
            raise ValueError("OOF and camera soil coverage must match training labels")
    oof = oof.set_index("sample_id").loc[train_ids].reset_index()
    for frame in (oof, cameras):
        actual = curve_array(truth.loc[frame["sample_id"]])
        frame["emd"] = [emd_score(y, p) for y, p in zip(actual, curve_array(frame), strict=True)]
    disagreements = [
        emd_score(a, b)
        for _, group in cameras.groupby("sample_id")
        for a, b in combinations(curve_array(group), 2)
    ]
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_blend": oof["emd"].to_numpy()})
    for name, frames in components.items():
        aligned = frames[0].set_index("sample_id").loc[train_ids]
        comparison[f"emd_{name}"] = [
            emd_score(y, p) for y, p in zip(curve_array(train), curve_array(aligned), strict=True)
        ]
        comparison[f"improvement_vs_{name}"] = comparison[f"emd_{name}"] - comparison["emd_blend"]

    candidate = _blend_predictions(components["multicrop"][2], components["nested_ridge"][2], ["sample_id"])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = curve_array(
        candidate.set_index("sample_id").loc[sample["sample_id"]]
    )
    validate_submission(submission, sample)
    experiment = "multicrop_nested_ridge_blend"
    summary = {
        "experiment": experiment, "weights": {"multicrop": 0.5, "nested_ridge": 0.5},
        "validation": "Fixed equal blend of saved outer leave-one-soil-out predictions; no weight selection",
        "loo_emd": float(oof["emd"].mean()),
        "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
        "camera_pairs": len(disagreements), "train_samples": len(train), "test_samples": len(sample),
        "sources": [{"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()} for path in sources],
    }
    oof.insert(0, "experiment", experiment)
    cameras.insert(0, "experiment", experiment)
    output = cfg.artifacts_dir / "experiments" / "model_blend"
    return {
        "oof": write_csv(oof, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(cameras, output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / f"{experiment}.csv"),
        "summary": write_json(summary, output / "summary.json"),
    }
