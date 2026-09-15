from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.io import write_csv, write_json
from soilgrain.multicrop import _score_rows
from soilgrain.pca_experiment import _camera_disagreement, submission_evidence
from soilgrain.photo_ridge import evaluate_photo_ridge, photo_ridge_curves
from soilgrain.physical_photo_experiment import _select_candidate
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.texture_experiment import _aligned_cache, _reference_difference


def write_spectral_photo_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    prior = cfg.artifacts_dir / "experiments" / "physical_photo"
    manifest_path = prior / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    spectrum_path = prior / "photo_spectrum_features.csv"
    if Path(manifest["feature_cache"]["path"]).resolve() != spectrum_path.resolve():
        raise ValueError("Spectral cache source changed from the recorded experiment")
    for item in [*manifest["sources"], *manifest["photo_sources"], *manifest["submissions"], manifest["feature_cache"]]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Recorded source changed: {item['path']}")
    truth, sample = [pd.read_csv(cfg.curated_file(key)) for key in ("train", "sample_submission")]
    validate_cumulative_curves(truth)
    for frame in (truth, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels/template require nonempty, unique, complete soil IDs")
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if len(train_ids) < 2 or set(train_ids) & set(test_ids):
        raise ValueError("Evaluation requires at least two training soils and disjoint test soils")
    keys = ["split", "sample_id", "camera", "path"]
    index_path = cfg.reports_dir / "photo_index.csv"
    index = pd.read_csv(index_path)[keys]
    if index[keys].isna().any().any() or index["path"].duplicated().any() or set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index requires unique paths, complete keys, and train/test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Photo index {split} coverage differs from labels/template")
    cache_path = cfg.artifacts_dir / "experiments" / "nested_texture_kernel" / "photo_features.csv"
    photos = _aligned_cache(cache_path, index, 17)
    spectrum = pd.read_csv(spectrum_path)
    spectrum_columns = [f"spectrum_{i}" for i in range(6)]
    if (set(c for c in spectrum if c.startswith("spectrum_")) != set(spectrum_columns)
            or spectrum[keys].isna().any().any() or spectrum["path"].duplicated().any()
            or not np.isfinite(spectrum[spectrum_columns].to_numpy()).all()):
        raise ValueError("Spectral cache requires complete unique keys and six finite measurements")
    expected = pd.MultiIndex.from_frame(index[keys])
    spectrum = spectrum.set_index(keys)
    if len(spectrum) != len(expected) or set(spectrum.index) != set(expected):
        raise ValueError("Spectral cache coverage differs from photo index")
    photos[[f"feature_{i}" for i in range(17, 23)]] = spectrum.loc[expected, spectrum_columns].to_numpy(dtype=float)
    columns = [f"feature_{i}" for i in range(23)]
    train_photos = photos.loc[photos["split"] == "train"]
    train = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test = photos.loc[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    curves = curve_array(truth)
    reference_paths = {"oof": prior / "oof_predictions.csv", "camera": prior / "camera_predictions.csv",
                       "submission": cfg.submissions_dir / "rgb_texture_spectral_ridge.csv"}
    saved = {name: pd.read_csv(path) for name, path in reference_paths.items()}
    for name in ("oof", "camera"):
        saved[name] = saved[name].loc[saved[name]["experiment"] == "spectral_ridge"]
    validate_submission(saved["submission"], sample)
    reference_oof, reference_views = evaluate_samples(train_photos, train_ids, curves, ridge_curves)
    reference = pd.DataFrame(reference_oof, columns=CANONICAL_GRAIN_LABELS)
    reference.insert(0, "sample_id", train_ids)
    reference_submission = sample.copy()
    reference_submission[ordered_grain_columns(sample)] = ridge_curves(train, curves, test)
    differences = {
        "oof": _reference_difference(reference, saved["oof"], ["sample_id"]),
        "camera": _reference_difference(reference_views, saved["camera"], ["sample_id", "camera"]),
        "submission": _reference_difference(reference_submission, saved["submission"], ["sample_id"]),
    }
    reference_errors = _score_rows(truth, reference)
    reference_camera, _pairs = _camera_disagreement(reference_views)
    print(f"Spectral incumbent reconstructed: {reference_errors.mean():.4f} EMD", flush=True)

    oof, views = evaluate_photo_ridge(train_photos, train_ids, curves)
    predicted = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    predicted.insert(0, "sample_id", train_ids)
    predicted["emd"] = _score_rows(truth, predicted)
    views["emd"] = _score_rows(truth, views)
    validate_cumulative_curves(predicted)
    validate_cumulative_curves(views)
    errors = predicted["emd"].to_numpy()
    camera_emd, pairs = _camera_disagreement(views)
    evidence = submission_evidence(errors, reference_errors, camera_emd, reference_camera)
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = photo_ridge_curves(train_photos, train_ids, curves, test)
    validate_submission(submission, sample)
    row = {"experiment": "spectral_photo_ridge", "loo_emd": float(errors.mean()), "paired_camera_disagreement_emd": camera_emd}
    selected = _select_candidate([row], float(reference_errors.mean()), reference_camera)
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_incumbent": reference_errors,
                               "emd_spectral_photo_ridge": errors, "improvement_emd": reference_errors - errors})
    output = cfg.artifacts_dir / "experiments" / "spectral_photo"
    paths = {
        "oof": write_csv(predicted, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(views, output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / "rgb_texture_spectral_photo_ridge.csv"),
    }
    sources = [Path(config_path), *(cfg.curated_file(k) for k in ("train", "sample_submission", "ppm")), index_path,
               manifest_path, cache_path, spectrum_path, *reference_paths.values(), Path("pyproject.toml"), Path("uv.lock")]
    paths["summary"] = write_json({
        **row, "feature_count": 23, "ridge_alpha": 10., "crop_mm": 100, "crop_pixels": 256,
        "spectral_bands_mm": manifest["spectral_bands_mm"],
        "photo_training": manifest["photo_training"], "validation": "Fixed whole-soil LOO; all held-out views excluded; no tuning or blend",
        "reference_experiment": "spectral_ridge", "train_samples": len(train_ids), "test_samples": len(test_ids),
        "photos": len(index), "camera_pairs": pairs,
        "incumbent_emd": float(reference_errors.mean()), "incumbent_camera_disagreement_emd": reference_camera,
        "reference_max_abs_differences": differences, "incumbent_screen_diagnostic": evidence,
        "submission_rule": "At most one if outer EMD and camera disagreement strictly improve versus spectral ridge; old four screens diagnostic only; no fallback",
        "selected_submission": str(paths["submission"]) if selected else None,
        "submission": {"path": str(paths["submission"]), "sha256": sha256(paths["submission"].read_bytes()).hexdigest()},
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
        "photo_sources": manifest["photo_sources"],
    }, output / "summary.json")
    print(f"Spectral photo ridge: {errors.mean():.4f} EMD; camera disagreement {camera_emd}; selected: {selected is not None}", flush=True)
    return paths
