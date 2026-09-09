from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.data import load_working_tables
from soilgrain.image_model import nearest_curves, photo_features
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def evaluate_samples(
    photos: pd.DataFrame, sample_ids: list[str], curves: np.ndarray, predictor: Callable
) -> tuple[np.ndarray, pd.DataFrame]:
    """Hold out a whole soil, then query its pooled and separate camera views."""
    columns = [c for c in photos if c.startswith("feature_")]
    features = photos.groupby("sample_id")[columns].mean().loc[sample_ids].to_numpy()
    oof, camera_rows = [], []
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(sample_ids)) != i
        oof.append(predictor(features[keep], curves[keep], features[i:i + 1])[0])
        views = photos[photos["sample_id"] == sample_id].groupby("camera")[columns].mean()
        predictions = predictor(features[keep], curves[keep], views.to_numpy())
        for camera, prediction in zip(views.index, predictions, strict=True):
            camera_rows.append({
                "sample_id": sample_id, "camera": camera,
                "emd": emd_score(curves[i], prediction),
                **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
            })
    return np.vstack(oof), pd.DataFrame(camera_rows)


def write_experiments(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    from soilgrain.ridge import ridge_curves

    cfg = load_config(config_path)
    train, _test, sample, ppm = load_working_tables(cfg)
    validate_cumulative_curves(train)
    index = pd.read_csv(cfg.reports_dir / "photo_index.csv")
    train_ids = train["sample_id"].astype(str).tolist()
    test_ids = sample["sample_id"].astype(str).tolist()
    curves = curve_array(train)
    cameras = ppm.set_index("phone")
    output = cfg.artifacts_dir / "experiments" / "first_batch"
    settings = [
        ("reference_rgb_100", "rgb", 100),
        ("gray_100", "gray", 100),
        ("normalized_gray_100", "normalized_gray", 100),
        ("gray_50", "gray", 50),
        ("gray_150", "gray", 150),
    ]
    summaries, all_oof, all_cameras, neighbor_rows = [], [], [], []
    reference_errors = None
    for name, color_mode, crop_mm in settings:
        values = np.vstack([
            photo_features(row.path, cameras.loc[row.camera], crop_mm=crop_mm, color_mode=color_mode)
            for row in index.itertuples(index=False)
        ])
        feature_columns = [f"feature_{i}" for i in range(values.shape[1])]
        photos = index[["split", "sample_id", "camera", "path"]].copy()
        photos[feature_columns] = values
        write_csv(photos, output / f"{name}_photo_features.csv")
        train_photos = photos[photos["split"] == "train"]
        x_train = train_photos.groupby("sample_id")[feature_columns].mean().loc[train_ids].to_numpy()
        x_test = photos[photos["split"] == "test"].groupby("sample_id")[feature_columns].mean().loc[test_ids].to_numpy()
        models = [(name, nearest_curves)]
        if name == "reference_rgb_100":
            models.append(("ridge_rgb_100", ridge_curves))
        for experiment, predictor in models:
            oof, camera_predictions = evaluate_samples(train_photos, train_ids, curves, predictor)
            predictions = predictor(x_train, curves, x_test)
            submission = sample.copy()
            submission[ordered_grain_columns(sample)] = predictions
            validate_submission(submission, sample)
            submission_path = write_csv(submission, cfg.submissions_dir / "experiments" / f"{experiment}.csv")
            oof_frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
            validate_cumulative_curves(oof_frame)
            errors = np.array([emd_score(y, p) for y, p in zip(curves, oof, strict=True)])
            if experiment == "reference_rgb_100":
                reference_errors = errors
            oof_frame.insert(0, "emd", errors)
            oof_frame.insert(0, "sample_id", train_ids)
            oof_frame.insert(0, "experiment", experiment)
            camera_predictions.insert(0, "experiment", experiment)
            all_oof.append(oof_frame)
            all_cameras.append(camera_predictions)

            disagreements = [
                emd_score(a, b)
                for _, group in camera_predictions.groupby("sample_id")
                for a, b in combinations(group[list(CANONICAL_GRAIN_LABELS)].to_numpy(), 2)
            ]
            h374_count = None
            if predictor is nearest_curves:
                scale = x_train.std(axis=0)
                scale[scale < 1e-12] = 1.0
                distances = np.square((x_test[:, None, :] - x_train[None, :, :]) / scale).sum(axis=2)
                neighbors = np.argsort(distances, axis=1, kind="stable")[:, :3]
                h374_count = int((np.asarray(train_ids)[neighbors] == "H374").any(axis=1).sum())
                for sample_id, selected in zip(test_ids, neighbors, strict=True):
                    for rank, neighbor in enumerate(selected, 1):
                        neighbor_rows.append({"experiment": experiment, "sample_id": sample_id, "rank": rank, "neighbor": train_ids[neighbor]})
            summaries.append({
                "experiment": experiment, "color_mode": color_mode, "crop_mm": crop_mm,
                "feature_count": values.shape[1], "ridge_alpha": 10.0 if predictor is ridge_curves else None,
                "loo_emd": emd_score(curves, oof),
                "samples_better_than_reference": int((errors < reference_errors - 1e-9).sum()),
                "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
                "camera_pairs": len(disagreements), "h374_test_neighbor_count": h374_count,
                "submission": str(submission_path),
            })
            print(f"{experiment}: {summaries[-1]['loo_emd']:.2f} EMD", flush=True)
    camera_predictions = pd.concat(all_cameras, ignore_index=True)
    camera_summary = camera_predictions.groupby(["experiment", "camera"])["emd"].agg(["mean", "count"]).reset_index()
    paths = {
        "summary": write_csv(pd.DataFrame(summaries), output / "summary.csv"),
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(camera_predictions, output / "camera_predictions.csv"),
        "camera_summary": write_csv(camera_summary, output / "camera_summary.csv"),
        "neighbors": write_csv(pd.DataFrame(neighbor_rows), output / "test_neighbors.csv"),
    }
    paths["manifest"] = write_json({
        "validation": "Leave one physical soil out, including every camera view; fit scaling within each fold",
        "selection": "Six predefined exploratory comparisons; no tuning or automatic winner selection",
        "reference_emd": float(np.mean(reference_errors)),
        "train_samples": len(train_ids), "test_samples": len(test_ids),
        "camera_diagnostic": "Prediction disagreement for paired cameras of held-out training soils; not unseen-iPhone performance",
    }, output / "manifest.json")
    return paths
