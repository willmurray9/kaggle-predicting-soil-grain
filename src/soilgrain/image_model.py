from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from soilgrain.baselines import leave_one_out_score
from soilgrain.config import load_config
from soilgrain.data import load_working_tables
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def photo_features(path: str | Path, camera: pd.Series) -> np.ndarray:
    """Color and texture of a central 100 mm square, rendered at 256 pixels."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    # PPM describes the native camera resolution; many training JPGs are smaller.
    ppm = float(camera["ppm"]) * max(image.size) / max(camera["width"], camera["height"])
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError(f"Invalid pixel scale for {path}: {ppm}.")
    side = round(100 * ppm)
    if side < 1 or side > min(image.size):
        raise ValueError(f"Cannot extract a 100 mm crop from {path} with PPM={ppm}.")
    left, top = (image.width - side) // 2, (image.height - side) // 2
    crop = image.crop((left, top, left + side, top + side)).resize((256, 256), Image.Resampling.LANCZOS)
    rgb = np.asarray(crop, dtype=float) / 255.0
    gray = rgb.mean(axis=2)
    color = np.quantile(rgb, [0.1, 0.5, 0.9], axis=(0, 1)).ravel()
    texture = [gray.std()]
    for lag in (1, 4, 16):
        texture.append((np.abs(gray[lag:] - gray[:-lag]).mean() + np.abs(gray[:, lag:] - gray[:, :-lag]).mean()) / 2)
    return np.concatenate([color, texture])


def sample_features(photo_index: pd.DataFrame, sample_ids: list[str], ppm: pd.DataFrame) -> np.ndarray:
    """Give each physical sample one feature vector, regardless of photo count."""
    cameras = ppm.set_index("phone")
    features = []
    for sample_id in sample_ids:
        photos = photo_index[photo_index["sample_id"] == sample_id]
        if photos.empty:
            raise ValueError(f"No photos for sample {sample_id}.")
        values = [photo_features(row.path, cameras.loc[row.camera]) for row in photos.itertuples(index=False)]
        features.append(np.mean(values, axis=0))
    return np.vstack(features)


def nearest_curves(train_features: np.ndarray, train_curves: np.ndarray, query_features: np.ndarray) -> np.ndarray:
    """Average three neighbors after scaling features using training samples only."""
    if len(train_features) == 0:
        raise ValueError("At least one training sample is required.")
    scale = train_features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    distances = np.square((query_features[:, None, :] - train_features[None, :, :]) / scale).sum(axis=2)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :3]
    return train_curves[neighbors].mean(axis=1)


def leave_one_out_predictions(features: np.ndarray, curves: np.ndarray) -> np.ndarray:
    if len(features) < 2:
        raise ValueError("Leave-one-sample-out validation requires at least two samples.")
    predictions = []
    for i in range(len(features)):
        keep = np.arange(len(features)) != i
        predictions.append(nearest_curves(features[keep], curves[keep], features[i:i + 1])[0])
    return np.vstack(predictions)


def write_image_model(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train, _test, sample, ppm = load_working_tables(cfg)
    validate_cumulative_curves(train)
    index = pd.read_csv(cfg.reports_dir / "photo_index.csv")
    train_ids = train["sample_id"].astype(str).tolist()
    test_ids = sample["sample_id"].astype(str).tolist()
    train_features = sample_features(index[index["split"] == "train"], train_ids, ppm)
    test_features = sample_features(index[index["split"] == "test"], test_ids, ppm)
    curves = curve_array(train)
    oof = leave_one_out_predictions(train_features, curves)
    predictions = nearest_curves(train_features, curves, test_features)

    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = predictions
    validate_submission(submission, sample)
    oof_frame = train[["sample_id"]].copy()
    oof_frame[ordered_grain_columns(train)] = oof
    validate_cumulative_curves(oof_frame)

    errors = pd.DataFrame({"sample_id": train_ids, "image_emd": [emd_score(y, p) for y, p in zip(curves, oof, strict=True)]})
    for method in ("mean", "median"):
        errors[f"{method}_emd"] = [
            emd_score(curves[i], getattr(np, method)(np.delete(curves, i, axis=0), axis=0))
            for i in range(len(curves))
        ]
    features = pd.DataFrame(np.vstack([train_features, test_features]), columns=[f"feature_{i}" for i in range(train_features.shape[1])])
    features.insert(0, "split", ["train"] * len(train_ids) + ["test"] * len(test_ids))
    features.insert(0, "sample_id", train_ids + test_ids)
    paths = {
        "submission": write_csv(submission, cfg.submissions_dir / "image_baseline.csv"),
        "oof": write_csv(oof_frame, cfg.reports_dir / "image_oof.csv"),
        "validation": write_csv(errors, cfg.reports_dir / "image_validation.csv"),
        "features": write_csv(features, cfg.reports_dir / "image_sample_features.csv"),
    }
    paths["summary"] = write_json(
        {
            "method": "Mean curve of 3 nearest samples in standardized color/texture features",
            "validation": "Leave one physical sample out; all its photos excluded; scaling fit on remaining samples",
            "crop_mm": 100,
            "crop_pixels": 256,
            "feature_count": int(train_features.shape[1]),
            "image_leave_one_out_score": emd_score(curves, oof),
            "mean_leave_one_out_score": leave_one_out_score(train, "mean"),
            "median_leave_one_out_score": leave_one_out_score(train, "median"),
            "train_samples": len(train_ids),
            "test_samples": len(test_ids),
            "paths": {key: str(value) for key, value in paths.items()},
        },
        cfg.reports_dir / "image_model_summary.json",
    )
    return paths
