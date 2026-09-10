from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.io import write_csv, write_json
from soilgrain.kernel_ridge import kernel_ridge_curves
from soilgrain.metrics import emd_score
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def write_kernel_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    first_batch = cfg.artifacts_dir / "experiments" / "first_batch"
    feature_path = first_batch / "reference_rgb_100_photo_features.csv"
    train = pd.read_csv(cfg.curated_file("train"))
    sample = pd.read_csv(cfg.curated_file("sample_submission"))
    validate_cumulative_curves(train)
    if train["sample_id"].duplicated().any():
        raise ValueError("Training labels contain duplicate soil IDs")
    train_ids, test_ids = train["sample_id"].tolist(), sample["sample_id"].tolist()
    curves = curve_array(train)
    photos = pd.read_csv(feature_path)
    columns = [f"feature_{i}" for i in range(13)]
    if not np.isfinite(photos[columns].to_numpy(dtype=float)).all():
        raise ValueError("Photo features must be finite")
    # Keep only the declared 13 features so the evaluator uses the same inputs.
    photos = photos[["split", "sample_id", "camera", *columns]]
    train_photos = photos[photos["split"] == "train"]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    oof, cameras = evaluate_samples(train_photos, train_ids, curves, kernel_ridge_curves)
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
    submission[ordered_grain_columns(sample)] = kernel_ridge_curves(train_features, curves, test_features)
    validate_submission(submission, sample)

    reference_paths = {
        "ridge": first_batch / "oof_predictions.csv",
        "multicrop": cfg.artifacts_dir / "experiments" / "multicrop" / "oof_predictions.csv",
    }
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_kernel": errors})
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

    experiment = "kernel_ridge_rgb_100"
    oof_frame.insert(0, "experiment", experiment)
    cameras.insert(0, "experiment", experiment)
    output = cfg.artifacts_dir / "experiments" / "kernel_ridge"
    paths = {
        "oof": write_csv(oof_frame, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(cameras, output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / f"{experiment}.csv"),
    }
    sources = [feature_path, cfg.curated_file("train"), cfg.curated_file("sample_submission"), *reference_paths.values()]
    paths["summary"] = write_json({
        "experiment": experiment, "kernel": "rbf", "alpha": 1.0, "gamma": 1.0 / 13,
        "feature_count": 13, "crop_mm": 100, "aggregation": "Equal mean across photos within each soil",
        "validation": "Leave one physical soil out; train-only scaling and kernel centering; unpenalized intercept",
        "selection": "One fixed configuration; no tuning or blends",
        "loo_emd": float(errors.mean()),
        "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
        "camera_pairs": len(disagreements), "train_samples": len(train_ids), "test_samples": len(test_ids),
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
    }, output / "summary.json")
    return paths
