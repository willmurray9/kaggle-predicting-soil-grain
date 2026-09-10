from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.image_model import SPATIAL_POSITIONS, nearest_curves, spatial_photo_features
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.multicrop import COMPONENTS, _score_rows, blend_predictions
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def write_spatial_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    """Compare five fixed crop positions with the submitted center-crop blend."""
    cfg = load_config(config_path)
    truth = pd.read_csv(cfg.curated_file("train"))
    sample = pd.read_csv(cfg.curated_file("sample_submission"))
    validate_cumulative_curves(truth)
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if len(train_ids) < 4 or len(set(train_ids)) != len(train_ids) or set(train_ids) & set(test_ids):
        raise ValueError("Require at least four unique training soils and disjoint test soils")
    index_path = cfg.reports_dir / "photo_index.csv"
    index = pd.read_csv(index_path)
    keys = ["split", "sample_id", "camera", "path"]
    if index[keys].isna().any().any() or index["path"].duplicated().any():
        raise ValueError("Photo index contains missing keys or duplicate paths")
    if set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index must contain only train and test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Photo index {split} soil coverage differs from labels/template")
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    if not cameras.index.is_unique:
        raise ValueError("Camera calibration contains duplicate phones")
    curves = curve_array(truth)
    output = cfg.artifacts_dir / "experiments" / "spatial_coverage"
    paths, oof_components, camera_components, submissions = {}, {}, {}, {}
    columns = [f"feature_{i}" for i in range(7)]

    for name, crop_mm in zip(COMPONENTS, (50, 100, 150), strict=True):
        photos = index[keys].copy()
        photos[columns] = np.vstack([
            spatial_photo_features(row.path, cameras.loc[row.camera], crop_mm=crop_mm)
            for row in index.itertuples(index=False)
        ])
        paths[f"{name}_features"] = write_csv(photos, output / f"{name}_photo_features.csv")
        train_photos = photos[photos["split"] == "train"]
        train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
        test_features = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
        oof, camera_predictions = evaluate_samples(train_photos, train_ids, curves, nearest_curves)
        oof_frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
        oof_frame.insert(0, "sample_id", train_ids)
        oof_components[name] = oof_frame
        camera_components[name] = camera_predictions
        submission = sample.copy()
        submission[ordered_grain_columns(sample)] = nearest_curves(train_features, curves, test_features)
        validate_submission(submission, sample)
        submissions[name] = submission
        print(f"spatial {name}: {emd_score(curves, oof):.4f} EMD", flush=True)

    blend_name = "spatial_multicrop"
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
        oof = (blended_oof if name == blend_name else oof_components[name]).copy()
        views = (blended_cameras if name == blend_name else camera_components[name]).copy()
        oof["emd"], views["emd"] = _score_rows(truth, oof), _score_rows(truth, views)
        disagreements = [
            emd_score(a, b)
            for _, group in views.groupby("sample_id")
            for a, b in combinations(curve_array(group), 2)
        ]
        summaries.append({
            "experiment": name, "loo_emd": float(oof["emd"].mean()),
            "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
            "camera_pairs": len(disagreements),
        })
        oof.insert(0, "experiment", name)
        views.insert(0, "experiment", name)
        all_oof.append(oof)
        all_cameras.append(views)

    reference_path = cfg.artifacts_dir / "experiments" / "multicrop" / "oof_predictions.csv"
    reference = pd.read_csv(reference_path)
    reference["emd"] = _score_rows(truth, reference)
    comparison = all_oof[-1][["sample_id", "emd"]].merge(
        reference[["sample_id", "emd"]], on="sample_id", how="outer",
        suffixes=("_spatial", "_reference"), validate="one_to_one",
    )
    if comparison.isna().any().any():
        raise ValueError("Reference and spatial OOF soil coverage differ")
    comparison = comparison.set_index("sample_id").loc[train_ids].reset_index()
    comparison["improvement_emd"] = comparison["emd_reference"] - comparison["emd_spatial"]
    paths.update({
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(pd.concat(all_cameras, ignore_index=True), output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / f"{blend_name}.csv"),
    })
    source_paths = [Path(config_path), index_path, cfg.curated_file("train"),
                    cfg.curated_file("sample_submission"), cfg.curated_file("ppm"), reference_path,
                    *map(Path, index["path"])]
    paths["summary"] = write_json({
        "positions": SPATIAL_POSITIONS,
        "position_definition": "Fractions of available crop-origin travel; floor to pixels after EXIF orientation",
        "aggregation": "Equal mean of five patch features per photo, then equal mean of photos per soil",
        "settings": "Grayscale 50/100/150 mm, 256 pixels, seven features, three uniform neighbors, equal curve blend",
        "validation": "Leave one physical soil out, excluding every photo and patch; fold-local scaling",
        "train_samples": len(train_ids), "test_samples": len(test_ids), "photos": len(index),
        "experiments": summaries,
        "submission_sha256": sha256(paths["submission"].read_bytes()).hexdigest(),
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in source_paths],
    }, output / "summary.json")
    print(f"{blend_name}: {summaries[-1]['loo_emd']:.4f} EMD", flush=True)
    return paths
