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
from soilgrain.metrics import emd_score
from soilgrain.multicrop import _score_rows
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.texture_features import TEXTURE_LAGS_PIXELS, TEXTURE_OFFSETS_MM, physical_texture_features


def _aligned_cache(path: Path, index: pd.DataFrame, count: int) -> pd.DataFrame:
    keys = ["split", "sample_id", "camera", "path"]
    columns = [f"feature_{i}" for i in range(count)]
    cache = pd.read_csv(path)
    if (set(c for c in cache if c.startswith("feature_")) != set(columns)
            or cache[keys].isna().any().any() or cache["path"].duplicated().any()):
        raise ValueError(f"Invalid feature cache keys or columns: {path}")
    if not np.isfinite(cache[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite feature cache: {path}")
    expected = pd.MultiIndex.from_frame(index[keys])
    cache = cache.set_index(keys)
    if len(cache) != len(expected) or len(cache.index.difference(expected)):
        raise ValueError(f"Feature cache coverage differs from photo index: {path}")
    return cache.loc[expected, columns].reset_index()


def _reference_difference(actual: pd.DataFrame, saved: pd.DataFrame, keys: list[str]) -> float:
    for frame in (actual, saved):
        validate_cumulative_curves(frame)
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("RGB reference contains missing or duplicate prediction keys")
    actual, saved = actual.set_index(keys), saved.set_index(keys)
    if len(actual) != len(saved) or len(actual.index.difference(saved.index)):
        raise ValueError("RGB reference prediction coverage differs")
    difference = float(np.max(np.abs(curve_array(actual) - curve_array(saved.loc[actual.index]))))
    if difference > 1e-10:
        raise ValueError(f"RGB reference predictions do not reconstruct: max difference {difference}")
    return difference


def write_texture_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    """Fixed ridge comparison of color and four physical texture measurements."""
    cfg = load_config(config_path)
    first = cfg.artifacts_dir / "experiments" / "first_batch"
    truth, sample = pd.read_csv(cfg.curated_file("train")), pd.read_csv(cfg.curated_file("sample_submission"))
    validate_cumulative_curves(truth)
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if (len(train_ids) < 2 or len(set(train_ids)) != len(train_ids)
            or len(set(test_ids)) != len(test_ids) or set(train_ids) & set(test_ids)
            or truth["sample_id"].isna().any() or sample["sample_id"].isna().any()):
        raise ValueError("Require unique, nonnull, disjoint train/test soils and at least two training soils")
    index_path = cfg.reports_dir / "photo_index.csv"
    keys = ["split", "sample_id", "camera", "path"]
    index = pd.read_csv(index_path)[keys]
    if index[keys].isna().any().any() or index["path"].duplicated().any():
        raise ValueError("Photo index has missing keys or duplicate paths")
    if set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index must contain only train and test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Photo index {split} coverage differs from labels/template")
    cache_paths = {"rgb": first / "reference_rgb_100_photo_features.csv", "gray": first / "gray_100_photo_features.csv"}
    caches = {color: _aligned_cache(cache_paths[color], index, count) for color, count in (("rgb", 13), ("gray", 7))}
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    if not cameras.index.is_unique:
        raise ValueError("Camera calibration contains duplicate phones")
    texture = np.vstack([physical_texture_features(row.path, cameras.loc[row.camera]) for row in index.itertuples(index=False)])
    if not np.isfinite(texture).all():
        raise ValueError("Physical texture features must be finite")
    texture_frame = index.copy()
    texture_frame[[f"texture_{i}" for i in range(4)]] = texture
    curves = curve_array(truth)
    reference_paths = {"oof": first / "oof_predictions.csv", "camera": first / "camera_predictions.csv",
                       "submission": cfg.submissions_dir / "experiments" / "ridge_rgb_100.csv"}
    saved = {name: pd.read_csv(path) for name, path in reference_paths.items()}
    for name in ("oof", "camera"):
        saved[name] = saved[name].loc[saved[name]["experiment"] == "ridge_rgb_100"]
    validate_submission(saved["submission"], sample)
    output = cfg.artifacts_dir / "experiments" / "physical_texture"
    paths, summaries, all_oof, all_cameras = {}, [], [], []
    comparison = truth[["sample_id"]].copy()
    settings = (("ridge_rgb_reference", "rgb", False), ("ridge_rgb_texture_100", "rgb", True),
                ("ridge_gray_100", "gray", False), ("ridge_gray_texture_100", "gray", True))
    reference_differences = {}
    for name, color, added_texture in settings:
        photos = caches[color].copy()
        base_count = 13 if color == "rgb" else 7
        if added_texture:
            photos[[f"feature_{i}" for i in range(base_count, base_count + 4)]] = texture
        columns = [c for c in photos if c.startswith("feature_")]
        train_photos = photos[photos["split"] == "train"]
        train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
        test_features = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
        oof, views = evaluate_samples(train_photos, train_ids, curves, ridge_curves)
        oof_frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
        oof_frame.insert(0, "sample_id", train_ids)
        oof_frame["emd"] = _score_rows(truth, oof_frame)
        validate_cumulative_curves(views)
        submission = sample.copy()
        submission[ordered_grain_columns(sample)] = ridge_curves(train_features, curves, test_features)
        validate_submission(submission, sample)
        if name == "ridge_rgb_reference":
            reference_differences = {
                "oof": _reference_difference(oof_frame, saved["oof"], ["sample_id"]),
                "camera": _reference_difference(views, saved["camera"], ["sample_id", "camera"]),
                "submission": _reference_difference(submission, saved["submission"], ["sample_id"]),
            }
        else:
            paths[f"{name}_submission"] = write_csv(submission, cfg.submissions_dir / f"{name}.csv")
        disagreements = [emd_score(a, b) for _, group in views.groupby("sample_id") for a, b in combinations(curve_array(group), 2)]
        summaries.append({
            "experiment": name, "color": color, "added_texture": added_texture, "feature_count": len(columns),
            "loo_emd": float(oof_frame["emd"].mean()), "camera_pairs": len(disagreements),
            "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
        })
        comparison[f"emd_{name}"] = oof_frame["emd"].to_numpy()
        oof_frame.insert(0, "experiment", name)
        views.insert(0, "experiment", name)
        all_oof.append(oof_frame)
        all_cameras.append(views)
        print(f"{name}: {summaries[-1]['loo_emd']:.4f} EMD", flush=True)

    for label, before, after in (
        ("texture_rgb", "ridge_rgb_reference", "ridge_rgb_texture_100"),
        ("texture_gray", "ridge_gray_100", "ridge_gray_texture_100"),
        ("remove_color", "ridge_rgb_reference", "ridge_gray_100"),
        ("remove_color_with_texture", "ridge_rgb_texture_100", "ridge_gray_texture_100"),
    ):
        comparison[f"improvement_{label}"] = comparison[f"emd_{before}"] - comparison[f"emd_{after}"]
    paths.update({
        "features": write_csv(texture_frame, output / "photo_texture_features.csv"),
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(pd.concat(all_cameras, ignore_index=True), output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
    })
    sources = [Path(config_path), index_path, cfg.curated_file("train"), cfg.curated_file("sample_submission"),
               cfg.curated_file("ppm"), *cache_paths.values(), *reference_paths.values(), *map(Path, index["path"])]
    paths["summary"] = write_json({
        "crop_mm": 100, "crop_pixels": 256, "nominal_offsets_mm": TEXTURE_OFFSETS_MM,
        "offsets_pixels": TEXTURE_LAGS_PIXELS, "actual_offsets_mm": [lag * 100 / 256 for lag in TEXTURE_LAGS_PIXELS],
        "texture": "Mean horizontal and vertical squared differences divided by twice grayscale variance; variance < 1e-12 gives zero",
        "ridge_alpha": 10., "aggregation": "Equal photo means per physical soil",
        "validation": "Leave one whole soil out, excluding all camera views; fold-fitted scaling and ridge",
        "selection": "Fixed 2x2 color/texture comparison; no parameter or weight search",
        "train_samples": len(train_ids), "test_samples": len(test_ids), "photos": len(index),
        "reference_max_abs_differences": reference_differences, "experiments": summaries,
        "submissions": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for key, p in paths.items() if key.endswith("_submission")],
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
    }, output / "summary.json")
    return paths
