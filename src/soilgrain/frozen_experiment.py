from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.frozen_features import extract_frozen_features
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def write_frozen_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train_path, sample_path, ppm_path = [cfg.curated_file(key) for key in ("train", "sample_submission", "ppm")]
    train, sample, ppm = [pd.read_csv(path) for path in (train_path, sample_path, ppm_path)]
    validate_cumulative_curves(train)
    if train["sample_id"].duplicated().any():
        raise ValueError("Training labels contain duplicate soil IDs")
    train_ids, test_ids = train["sample_id"].tolist(), sample["sample_id"].tolist()
    curves = curve_array(train)
    index_path = cfg.reports_dir / "photo_index.csv"
    index = pd.read_csv(index_path)
    photos, encoder = extract_frozen_features(index, ppm)
    columns = [f"feature_{i}" for i in range(512)]
    if not np.isfinite(photos[columns].to_numpy()).all():
        raise ValueError("Frozen image features must be finite")
    train_photos = photos[photos["split"] == "train"]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    oof, cameras = evaluate_samples(train_photos, train_ids, curves, ridge_curves)
    errors = np.array([emd_score(y, p) for y, p in zip(curves, oof, strict=True)])
    oof_frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    oof_frame.insert(0, "sample_id", train_ids)
    oof_frame["emd"] = errors
    validate_cumulative_curves(oof_frame)
    validate_cumulative_curves(cameras)
    disagreements = [
        emd_score(a, b)
        for _, group in cameras.groupby("sample_id")
        for a, b in combinations(curve_array(group), 2)
    ]
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = ridge_curves(train_features, curves, test_features)
    validate_submission(submission, sample)

    reference_paths = {
        "ridge": cfg.artifacts_dir / "experiments" / "first_batch" / "oof_predictions.csv",
        "multicrop": cfg.artifacts_dir / "experiments" / "multicrop" / "oof_predictions.csv",
    }
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_frozen": errors})
    for name, path in reference_paths.items():
        reference = pd.read_csv(path)
        if name == "ridge":
            reference = reference[reference["experiment"] == "ridge_rgb_100"]
        if reference["sample_id"].duplicated().any() or set(reference["sample_id"]) != set(train_ids):
            raise ValueError(f"{name} OOF soil coverage differs from training labels")
        aligned = reference.set_index("sample_id").loc[train_ids].reset_index()
        validate_cumulative_curves(aligned)
        comparison[f"emd_{name}"] = [
            emd_score(y, p) for y, p in zip(curves, curve_array(aligned), strict=True)
        ]
        comparison[f"improvement_vs_{name}"] = comparison[f"emd_{name}"] - errors

    experiment = "frozen_resnet18_ridge"
    oof_frame.insert(0, "experiment", experiment)
    cameras.insert(0, "experiment", experiment)
    output = cfg.artifacts_dir / "experiments" / "frozen_resnet18"
    paths = {
        "features": write_csv(photos, output / "photo_features.csv"),
        "oof": write_csv(oof_frame, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(cameras, output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / f"{experiment}.csv"),
    }
    sources = [Path(config_path), train_path, sample_path, ppm_path, index_path, *reference_paths.values()]
    paths["summary"] = write_json({
        "experiment": experiment, "encoder": encoder, "feature_count": 512, "ridge_alpha": 10.0,
        "aggregation": "Equal mean across photos within each soil",
        "validation": "Leave one physical soil out; frozen encoder; train-only ridge scaling and fitting",
        "selection": "One fixed encoder and ridge setting; no tuning or blends",
        "loo_emd": float(errors.mean()),
        "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
        "camera_pairs": len(disagreements), "train_samples": len(train_ids), "test_samples": len(test_ids),
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
        "photo_sources": [{"path": p, "sha256": sha256(Path(p).read_bytes()).hexdigest()} for p in index["path"]],
    }, output / "summary.json")
    return paths
