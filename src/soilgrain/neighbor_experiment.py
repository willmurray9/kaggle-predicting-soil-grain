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
from soilgrain.multicrop import COMPONENTS
from soilgrain.nested_neighbors import (
    NEIGHBOR_COUNTS, evaluate_nested_neighbors, predict_multicrop_neighbors, select_neighbor_count,
)
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def write_neighbor_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train_path, sample_path = [cfg.curated_file(key) for key in ("train", "sample_submission")]
    truth, sample = [pd.read_csv(path) for path in (train_path, sample_path)]
    validate_cumulative_curves(truth)
    if truth["sample_id"].duplicated().any() or sample["sample_id"].duplicated().any():
        raise ValueError("Labels and submission template must contain unique soil IDs")
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    curves = curve_array(truth)
    first = cfg.artifacts_dir / "experiments" / "first_batch"
    sources = {name: first / f"{name}_photo_features.csv" for name in COMPONENTS}
    columns = [f"feature_{i}" for i in range(7)]
    train_photos, train_features, test_features = {}, {}, {}
    photo_keys = None
    for name, source in sources.items():
        photos = pd.read_csv(source)
        keys = ["split", "sample_id", "camera", "path"]
        if photos[keys].isna().any().any() or photos["path"].duplicated().any():
            raise ValueError("Photo features require unique paths and complete soil/camera identifiers")
        if set(photos["split"]) != {"train", "test"} or not np.isfinite(photos[columns].to_numpy()).all():
            raise ValueError("Photo features must be finite and have train/test splits")
        current_keys = set(photos[keys].itertuples(index=False, name=None))
        if photo_keys is not None and current_keys != photo_keys:
            raise ValueError("Crop caches must cover the same photos, soils, and cameras")
        photo_keys = current_keys
        for split, ids in (("train", train_ids), ("test", test_ids)):
            if set(photos.loc[photos["split"] == split, "sample_id"]) != set(ids):
                raise ValueError(f"Cached {split} soil coverage differs from labels/template")
        train_photos[name] = photos.loc[photos["split"] == "train", [*keys, *columns]]
        train_features[name] = train_photos[name].groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
        test_features[name] = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()

    oof, cameras, outer_selection = evaluate_nested_neighbors(train_photos, train_ids, curves)
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
    selected, scores = select_neighbor_count(train_features, curves)
    final_selection = pd.DataFrame([
        {"n_neighbors": count, "loo_selection_emd": score, "selected": count == selected}
        for count, score in scores.items()
    ])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = predict_multicrop_neighbors(
        train_features, curves, test_features, n_neighbors=selected,
    )
    validate_submission(submission, sample)

    reference_path = cfg.artifacts_dir / "experiments" / "multicrop" / "oof_predictions.csv"
    reference = pd.read_csv(reference_path)
    if reference["sample_id"].duplicated().any() or set(reference["sample_id"]) != set(train_ids):
        raise ValueError("Reference OOF soil coverage differs from training labels")
    reference = reference.set_index("sample_id").loc[train_ids].reset_index()
    validate_cumulative_curves(reference)
    reference_errors = np.array([emd_score(y, p) for y, p in zip(curves, curve_array(reference), strict=True)])
    comparison = pd.DataFrame({
        "sample_id": train_ids, "emd_nested": errors, "emd_fixed3": reference_errors,
        "improvement_vs_fixed3": reference_errors - errors,
    })
    experiment = "nested_neighbors_multicrop"
    oof_frame.insert(0, "experiment", experiment)
    cameras.insert(0, "experiment", experiment)
    output = cfg.artifacts_dir / "experiments" / "nested_neighbors"
    paths = {
        "oof": write_csv(oof_frame, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(cameras, output / "camera_predictions.csv"),
        "outer_selection": write_csv(outer_selection, output / "outer_selection.csv"),
        "final_selection": write_csv(final_selection, output / "final_selection.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / f"{experiment}.csv"),
    }
    source_paths = [Path(config_path), train_path, sample_path, *sources.values(), first / "manifest.json", reference_path]
    paths["summary"] = write_json({
        "experiment": experiment, "components": list(COMPONENTS), "neighbor_counts": list(NEIGHBOR_COUNTS),
        "final_n_neighbors": selected,
        "aggregation": "Equal photo means; one shared neighbor count; uniform neighbors; equal crop prediction blend",
        "validation": "Nested leave-one-soil-out; select count only on outer training soils; refit scaling in every inner fold",
        "selection": "Minimum mean inner EMD over fixed grid; exact ties prefer more neighbors",
        "final_selection_note": "Full-training LOO scores choose the final count; they are not validation estimates",
        "outer_loo_emd": float(errors.mean()),
        "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
        "camera_pairs": len(disagreements), "train_samples": len(train_ids), "test_samples": len(test_ids),
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in source_paths],
    }, output / "summary.json")
    return paths
