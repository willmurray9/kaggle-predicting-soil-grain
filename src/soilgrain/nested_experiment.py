from __future__ import annotations

import json
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.nested_ridge import RIDGE_ALPHAS, evaluate_nested_ridge, select_ridge_alpha
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def write_nested_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train_path, sample_path = [cfg.curated_file(key) for key in ("train", "sample_submission")]
    train, sample = [pd.read_csv(path) for path in (train_path, sample_path)]
    validate_cumulative_curves(train)
    if train["sample_id"].duplicated().any() or sample["sample_id"].duplicated().any():
        raise ValueError("Labels and submission template must contain unique soil IDs")
    train_ids, test_ids = train["sample_id"].tolist(), sample["sample_id"].tolist()
    curves = curve_array(train)
    frozen = cfg.artifacts_dir / "experiments" / "frozen_resnet18"
    cache_path, manifest_path = frozen / "photo_features.csv", frozen / "summary.json"
    columns = [f"feature_{i}" for i in range(512)]
    # Preserve the encoder's float32 photo aggregation when reloading its CSV.
    photos = pd.read_csv(cache_path, dtype={column: np.float32 for column in columns})
    encoder = json.loads(manifest_path.read_text())["encoder"]
    if photos[["split", "sample_id", "camera", "path"]].isna().any().any() or photos["path"].duplicated().any():
        raise ValueError("Cached photos must have unique paths and complete soil/camera identifiers")
    if set(photos["split"]) != {"train", "test"} or not np.isfinite(photos[columns].to_numpy()).all():
        raise ValueError("Cached features must be finite and have train/test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(photos.loc[photos["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Cached {split} soil coverage differs from labels/template")
    train_photos = photos[photos["split"] == "train"]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()

    oof, cameras, outer_selection = evaluate_nested_ridge(train_photos, train_ids, curves)
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
    final_alpha, scores = select_ridge_alpha(train_features, curves)
    final_selection = pd.DataFrame([
        {"alpha": alpha, "loo_selection_emd": score, "selected": alpha == final_alpha}
        for alpha, score in scores.items()
    ])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = ridge_curves(
        train_features, curves, test_features, alpha=final_alpha,
    )
    validate_submission(submission, sample)

    reference_paths = {
        "frozen": frozen / "oof_predictions.csv",
        "multicrop": cfg.artifacts_dir / "experiments" / "multicrop" / "oof_predictions.csv",
    }
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_nested": errors})
    for name, path in reference_paths.items():
        reference = pd.read_csv(path)
        if reference["sample_id"].duplicated().any() or set(reference["sample_id"]) != set(train_ids):
            raise ValueError(f"{name} OOF soil coverage differs from training labels")
        aligned = reference.set_index("sample_id").loc[train_ids].reset_index()
        validate_cumulative_curves(aligned)
        comparison[f"emd_{name}"] = [
            emd_score(y, p) for y, p in zip(curves, curve_array(aligned), strict=True)
        ]
        comparison[f"improvement_vs_{name}"] = comparison[f"emd_{name}"] - errors

    experiment = "frozen_resnet18_nested_ridge"
    oof_frame.insert(0, "experiment", experiment)
    cameras.insert(0, "experiment", experiment)
    output = cfg.artifacts_dir / "experiments" / "nested_ridge"
    paths = {
        "oof": write_csv(oof_frame, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(cameras, output / "camera_predictions.csv"),
        "outer_selection": write_csv(outer_selection, output / "outer_selection.csv"),
        "final_selection": write_csv(final_selection, output / "final_selection.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / f"{experiment}.csv"),
    }
    sources = [Path(config_path), train_path, sample_path, cache_path, manifest_path, *reference_paths.values()]
    paths["summary"] = write_json({
        "experiment": experiment, "encoder": encoder, "feature_count": 512,
        "ridge_alphas": list(RIDGE_ALPHAS), "final_alpha": final_alpha,
        "aggregation": "Equal mean across photos within each soil",
        "validation": "Nested leave-one-soil-out; select alpha only on outer training soils; refit scaling in every inner fold",
        "selection": "Minimum mean inner EMD over fixed grid; exact ties prefer larger alpha",
        "final_selection_note": "Full-training LOO scores select the final alpha; they are not validation estimates",
        "outer_loo_emd": float(errors.mean()),
        "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
        "camera_pairs": len(disagreements), "train_samples": len(train_ids), "test_samples": len(test_ids),
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
    }, output / "summary.json")
    return paths
