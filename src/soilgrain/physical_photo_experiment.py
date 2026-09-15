from __future__ import annotations

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.io import write_csv, write_json
from soilgrain.multicrop import _score_rows
from soilgrain.pca_experiment import _camera_disagreement, submission_evidence
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.texture_experiment import _aligned_cache, _reference_difference
from soilgrain.photo_ridge import evaluate_photo_ridge, photo_ridge_curves
from soilgrain.spectral_features import SPECTRAL_BANDS_MM, physical_spectrum_features


def _select_candidate(rows, reference_emd, reference_camera):
    eligible = [row for row in rows if row["paired_camera_disagreement_emd"] is not None and reference_camera is not None
                and row["loo_emd"] < reference_emd and row["paired_camera_disagreement_emd"] < reference_camera]
    return min(eligible, key=lambda row: (row["loo_emd"], row["experiment"] != "spectral_ridge"))["experiment"] if eligible else None


def write_physical_photo_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    truth, sample = [pd.read_csv(cfg.curated_file(key)) for key in ("train", "sample_submission")]
    validate_cumulative_curves(truth)
    for frame in (truth, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels/template require nonempty, unique, complete soil IDs")
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if len(train_ids) < 2 or set(train_ids) & set(test_ids):
        raise ValueError("Evaluation requires at least two training soils and disjoint test soils")
    index_path = cfg.reports_dir / "photo_index.csv"
    keys = ["split", "sample_id", "camera", "path"]
    index = pd.read_csv(index_path)[keys]
    if index[keys].isna().any().any() or index["path"].duplicated().any() or set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index requires unique paths, complete keys, and train/test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Photo index {split} coverage differs from labels/template")
    cache_dir = cfg.artifacts_dir / "experiments" / "nested_texture_kernel"
    cache_path, manifest_path = cache_dir / "photo_features.csv", cache_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["feature_cache"]["path"]).resolve() != cache_path.resolve():
        raise ValueError("Feature-cache source changed from the recorded experiment")
    for item in [*manifest["sources"], manifest["feature_cache"]]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Recorded source changed: {item['path']}")
    texture_manifest_path = cfg.artifacts_dir / "experiments" / "physical_texture" / "summary.json"
    texture_manifest = json.loads(texture_manifest_path.read_text())
    for item in texture_manifest["sources"]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Recorded source changed: {item['path']}")
    ppm = pd.read_csv(cfg.curated_file("ppm"))
    if ppm["phone"].isna().any() or ppm["phone"].duplicated().any():
        raise ValueError("Camera calibration requires complete, unique phones")
    cameras = ppm.set_index("phone")
    photo_sources = [{"path": p, "sha256": sha256(Path(p).read_bytes()).hexdigest()} for p in index["path"]]
    photos = _aligned_cache(cache_path, index, 17)
    columns = [f"feature_{i}" for i in range(17)]
    train_photos = photos.loc[photos["split"] == "train"]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos.loc[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    curves = curve_array(truth)
    incumbent_dir = cfg.artifacts_dir / "experiments" / "physical_texture"
    reference_paths = {"oof": incumbent_dir / "oof_predictions.csv", "camera": incumbent_dir / "camera_predictions.csv",
                       "submission": cfg.submissions_dir / "ridge_rgb_texture_100.csv"}
    saved = {name: pd.read_csv(path) for name, path in reference_paths.items()}
    for name in ("oof", "camera"):
        saved[name] = saved[name].loc[saved[name]["experiment"] == "ridge_rgb_texture_100"]
    validate_submission(saved["submission"], sample)
    reference_oof, reference_views = evaluate_samples(train_photos, train_ids, curves, ridge_curves)
    reference = pd.DataFrame(reference_oof, columns=CANONICAL_GRAIN_LABELS)
    reference.insert(0, "sample_id", train_ids)
    reference_submission = sample.copy()
    reference_submission[ordered_grain_columns(sample)] = ridge_curves(train_features, curves, test_features)
    differences = {
        "oof": _reference_difference(reference, saved["oof"], ["sample_id"]),
        "camera": _reference_difference(reference_views, saved["camera"], ["sample_id", "camera"]),
        "submission": _reference_difference(reference_submission, saved["submission"], ["sample_id"]),
    }
    reference_errors = _score_rows(truth, reference)
    reference_camera, _pairs = _camera_disagreement(reference_views)
    print(f"RGB + texture incumbent reconstructed: {reference_errors.mean():.4f} EMD", flush=True)

    spectral = np.vstack([physical_spectrum_features(row.path, cameras.loc[row.camera]) for row in index.itertuples(index=False)])
    if spectral.shape != (len(index), 6) or not np.isfinite(spectral).all():
        raise ValueError("Spectral features must contain six finite values per photo")
    spectral_frame = pd.concat([index.reset_index(drop=True), pd.DataFrame(spectral, columns=[f"spectrum_{i}" for i in range(6)])], axis=1)
    augmented = photos.copy()
    augmented[[f"feature_{i}" for i in range(17, 23)]] = spectral
    output = cfg.artifacts_dir / "experiments" / "physical_photo"
    paths, summaries, all_oof, all_views = {}, [], [], []
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_incumbent": reference_errors})
    for name, feature_frame in (("spectral_ridge", augmented), ("photo_ridge", photos)):
        model_columns = [c for c in feature_frame if c.startswith("feature_")]
        model_photos = feature_frame.loc[feature_frame["split"] == "train"]
        model_train = model_photos.groupby("sample_id")[model_columns].mean().loc[train_ids].to_numpy()
        model_test = feature_frame.loc[feature_frame["split"] == "test"].groupby("sample_id")[model_columns].mean().loc[test_ids].to_numpy()
        if name == "spectral_ridge":
            oof, views = evaluate_samples(model_photos, train_ids, curves, ridge_curves)
            test_predictions = ridge_curves(model_train, curves, model_test)
        else:
            oof, views = evaluate_photo_ridge(model_photos, train_ids, curves)
            test_predictions = photo_ridge_curves(model_photos, train_ids, curves, model_test)
        predicted = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
        predicted.insert(0, "sample_id", train_ids)
        predicted["emd"] = _score_rows(truth, predicted)
        views["emd"] = _score_rows(truth, views)
        validate_cumulative_curves(predicted)
        validate_cumulative_curves(views)
        submission = sample.copy()
        submission[ordered_grain_columns(sample)] = test_predictions
        validate_submission(submission, sample)
        errors = predicted["emd"].to_numpy()
        camera_emd, pairs = _camera_disagreement(views)
        evidence = submission_evidence(errors, reference_errors, camera_emd, reference_camera)
        summaries.append({"experiment": name, "feature_count": len(model_columns), "loo_emd": float(errors.mean()),
                          "paired_camera_disagreement_emd": camera_emd, "camera_pairs": pairs,
                          "incumbent_screen_diagnostic": evidence})
        comparison[f"emd_{name}"] = errors
        comparison[f"improvement_{name}"] = reference_errors - errors
        predicted.insert(0, "experiment", name)
        views.insert(0, "experiment", name)
        all_oof.append(predicted)
        all_views.append(views)
        paths[f"{name}_submission"] = write_csv(submission, cfg.submissions_dir / f"rgb_texture_{name}.csv")
        print(f"{name}: {errors.mean():.4f} EMD; camera disagreement {camera_emd}", flush=True)
    paths.update({
        "features": write_csv(spectral_frame, output / "photo_spectrum_features.csv"),
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(pd.concat(all_views, ignore_index=True), output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
    })
    selected = _select_candidate(summaries, float(reference_errors.mean()), reference_camera)
    sources = [Path(config_path), *(cfg.curated_file(k) for k in ("train", "sample_submission", "ppm")), index_path,
               cache_path, manifest_path, texture_manifest_path, *reference_paths.values(), Path("pyproject.toml"), Path("uv.lock")]
    paths["summary"] = write_json({
        "ridge_alpha": 10., "crop_mm": 100, "crop_pixels": 256,
        "spectral_bands_mm": SPECTRAL_BANDS_MM,
        "spectrum": "Hann-weighted mean subtraction and window; FFT2 radial power fractions at image wavelengths, not grain diameters; total power <=1e-12 gives zero",
        "spectral_aggregation": "Equal float64 photo means per soil; six spectral fractions appended to existing 17 features",
        "photo_training": "Weight 1/photos-in-soil; scaling fitted to training-soil means; unpenalized intercept; query mean before curve repair",
        "validation": "Fixed whole-soil LOO; all views of held-out soil excluded; no tuning or blend",
        "train_samples": len(train_ids), "test_samples": len(test_ids), "photos": len(index),
        "incumbent_emd": float(reference_errors.mean()), "incumbent_camera_disagreement_emd": reference_camera,
        "reference_max_abs_differences": differences, "experiments": summaries,
        "submission_rule": "At most one with strictly lower outer EMD and camera disagreement than incumbent; lower EMD wins, exact tie spectral; old four screens diagnostic only",
        "selected_submission": str(paths[f"{selected}_submission"]) if selected else None,
        "submissions": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for key, p in paths.items() if key.endswith("_submission")],
        "feature_cache": {"path": str(paths["features"]), "sha256": sha256(paths["features"].read_bytes()).hexdigest()},
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources], "photo_sources": photo_sources,
        "versions": {"python": python_version(), **{name: version(name) for name in ("numpy", "pandas", "pillow")}},
    }, output / "summary.json")
    return paths
