from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.image_model import nearest_curves
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.multicrop import COMPONENTS, blend_predictions
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def balance_cameras(photos: pd.DataFrame) -> pd.DataFrame:
    """One row per camera, so subsequent soil means weight cameras equally."""
    keys = ["split", "sample_id", "camera"]
    columns = [c for c in photos if c.startswith("feature_")]
    if photos.empty or not columns or photos[keys].isna().any().any():
        raise ValueError("Photo features require nonempty data and complete split, soil, and camera keys")
    if not np.isfinite(photos[columns].to_numpy(dtype=float)).all():
        raise ValueError("Photo features must be finite")
    return photos.groupby(keys, as_index=False)[columns].mean()


def _score_predictions(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    validate_cumulative_curves(predictions)
    actual = truth.set_index("sample_id").loc[predictions["sample_id"]].reset_index()
    scored = predictions.copy()
    scored["emd"] = [
        emd_score(y, p) for y, p in zip(curve_array(actual), curve_array(scored), strict=True)
    ]
    return scored


def write_camera_balance(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    first_batch = cfg.artifacts_dir / "experiments" / "first_batch"
    sources = {name: first_batch / f"{name}_photo_features.csv" for name in COMPONENTS}
    truth = pd.read_csv(cfg.curated_file("train"))
    sample = pd.read_csv(cfg.curated_file("sample_submission"))
    validate_cumulative_curves(truth)
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    curves = curve_array(truth)
    oof_components, camera_components, submissions = {}, {}, {}

    for name, source in sources.items():
        photos = balance_cameras(pd.read_csv(source))
        columns = [c for c in photos if c.startswith("feature_")]
        train_photos = photos[photos["split"] == "train"]
        train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
        test_features = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
        oof, cameras = evaluate_samples(train_photos, train_ids, curves, nearest_curves)
        oof_frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
        oof_frame.insert(0, "sample_id", train_ids)
        oof_components[name] = oof_frame
        camera_components[name] = cameras
        submission = sample.copy()
        submission[ordered_grain_columns(sample)] = nearest_curves(train_features, curves, test_features)
        validate_submission(submission, sample)
        submissions[name] = submission

    blend_name = "camera_balanced_multicrop"
    blended_oof = blend_predictions(oof_components, ["sample_id"])
    blended_cameras = blend_predictions(camera_components, ["sample_id", "camera"])
    blended_submission = blend_predictions(submissions, ["sample_id"])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = curve_array(
        blended_submission.set_index("sample_id").loc[test_ids].reset_index()
    )
    validate_submission(submission, sample)

    all_oof, all_cameras, summaries = [], [], []
    for name in (*COMPONENTS, blend_name):
        oof = _score_predictions(blended_oof if name == blend_name else oof_components[name], truth)
        cameras = _score_predictions(blended_cameras if name == blend_name else camera_components[name], truth)
        disagreements = [
            emd_score(a, b)
            for _, group in cameras.groupby("sample_id")
            for a, b in combinations(curve_array(group), 2)
        ]
        summaries.append({
            "experiment": name, "loo_emd": float(oof["emd"].mean()),
            "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
            "camera_pairs": len(disagreements),
        })
        oof.insert(0, "experiment", name)
        cameras.insert(0, "experiment", name)
        all_oof.append(oof)
        all_cameras.append(cameras)

    reference_path = cfg.artifacts_dir / "experiments" / "multicrop" / "oof_predictions.csv"
    reference = _score_predictions(pd.read_csv(reference_path), truth)
    comparison = all_oof[-1][["sample_id", "emd"]].merge(
        reference[["sample_id", "emd"]], on="sample_id", how="outer",
        suffixes=("_balanced", "_reference"), validate="one_to_one",
    )
    if comparison.isna().any().any():
        raise ValueError("Reference and balanced OOF soil coverage differ")
    comparison["improvement_emd"] = comparison["emd_reference"] - comparison["emd_balanced"]

    output = cfg.artifacts_dir / "experiments" / "camera_balance"
    paths = {
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(pd.concat(all_cameras, ignore_index=True), output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / f"{blend_name}.csv"),
    }
    source_paths = [*sources.values(), cfg.curated_file("train"), cfg.curated_file("sample_submission"), reference_path]
    paths["summary"] = write_json({
        "aggregation": "Equal mean within each camera, then equal mean across cameras for each soil",
        "validation": "Leave one physical soil out, excluding every camera view; fold-local feature scaling",
        "settings": "Existing grayscale 50/100/150 mm features, 3-NN, equal prediction blend; no tuning",
        "train_samples": len(train_ids), "test_samples": len(test_ids),
        "experiments": summaries,
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in source_paths],
    }, output / "summary.json")
    return paths
