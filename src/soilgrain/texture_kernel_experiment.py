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
from soilgrain.kernel_ridge import kernel_ridge_curves
from soilgrain.multicrop import _score_rows
from soilgrain.nested_kernel import KERNEL_SETTINGS, evaluate_nested_kernel, select_kernel_parameters
from soilgrain.pca_experiment import _camera_disagreement, submission_evidence
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.texture_experiment import _aligned_cache, _reference_difference


def _texture_photos(rgb_path, texture_path, index):
    keys, columns = ["split", "sample_id", "camera", "path"], [f"texture_{i}" for i in range(4)]
    rgb = _aligned_cache(rgb_path, index, 13)
    texture = pd.read_csv(texture_path)
    if (set(c for c in texture if c.startswith("texture_")) != set(columns)
            or texture[keys].isna().any().any() or texture["path"].duplicated().any()
            or not np.isfinite(texture[columns].to_numpy()).all()):
        raise ValueError("Texture cache requires complete unique keys and four finite measurements")
    expected = pd.MultiIndex.from_frame(index[keys])
    texture = texture.set_index(keys)
    if len(texture) != len(expected) or set(texture.index) != set(expected):
        raise ValueError("Texture cache coverage differs from photo index")
    values = texture.loc[expected, columns].to_numpy()
    return pd.concat([rgb, pd.DataFrame(values, columns=[f"feature_{i}" for i in range(13, 17)])], axis=1)


def write_texture_kernel_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    truth, sample = [pd.read_csv(cfg.curated_file(key)) for key in ("train", "sample_submission")]
    validate_cumulative_curves(truth)
    for frame in (truth, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels/template require nonempty, unique, complete soil IDs")
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if len(train_ids) < 3 or set(train_ids) & set(test_ids):
        raise ValueError("Nested evaluation requires at least three training soils and disjoint test soils")
    index_path = cfg.reports_dir / "photo_index.csv"
    keys = ["split", "sample_id", "camera", "path"]
    index = pd.read_csv(index_path)[keys]
    if index[keys].isna().any().any() or index["path"].duplicated().any() or set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index requires unique paths, complete keys, and train/test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Photo index {split} coverage differs from labels/template")
    incumbent_dir = cfg.artifacts_dir / "experiments" / "physical_texture"
    manifest_path = incumbent_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    for item in [*manifest["sources"], *manifest.get("submissions", [])]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Incumbent source changed: {item['path']}")
    rgb_path = cfg.artifacts_dir / "experiments" / "first_batch" / "reference_rgb_100_photo_features.csv"
    texture_path = incumbent_dir / "photo_texture_features.csv"
    photos = _texture_photos(rgb_path, texture_path, index)
    columns = [f"feature_{i}" for i in range(17)]
    train_photos = photos.loc[photos["split"] == "train"]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos.loc[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    curves = curve_array(truth)
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

    fixed_oof, fixed_views = evaluate_samples(train_photos, train_ids, curves, kernel_ridge_curves)
    nested_oof, nested_views, outer_selection = evaluate_nested_kernel(train_photos, train_ids, curves)
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_incumbent": reference_errors})
    summaries, all_oof, all_views = [], [], []
    for name, oof, views in (("fixed", fixed_oof, fixed_views), ("nested", nested_oof, nested_views)):
        frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
        frame.insert(0, "sample_id", train_ids)
        frame["emd"] = _score_rows(truth, frame)
        views["emd"] = _score_rows(truth, views)
        camera_emd, pairs = _camera_disagreement(views)
        errors = frame["emd"].to_numpy()
        summaries.append({"experiment": name, "outer_loo_emd": float(errors.mean()),
                          "paired_camera_disagreement_emd": camera_emd, "camera_pairs": pairs})
        comparison[f"emd_{name}"] = errors
        comparison[f"improvement_{name}_vs_incumbent"] = reference_errors - errors
        if name == "nested":
            evidence = submission_evidence(errors, reference_errors, camera_emd, reference_camera)
        frame.insert(0, "experiment", name)
        views.insert(0, "experiment", name)
        all_oof.append(frame)
        all_views.append(views)
        print(f"{name} kernel: {errors.mean():.4f} EMD; camera disagreement {camera_emd}", flush=True)
    comparison["improvement_nested_vs_fixed"] = comparison["emd_fixed"] - comparison["emd_nested"]
    selected, scores = select_kernel_parameters(train_features, curves)
    alpha, gamma_scale = selected
    final_selection = pd.DataFrame([{"alpha": a, "gamma_scale": g, "loo_selection_emd": score, "selected": (a, g) == selected}
                                    for (a, g), score in scores.items()])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = kernel_ridge_curves(train_features, curves, test_features, alpha=alpha, gamma=gamma_scale / 17)
    validate_submission(submission, sample)
    output = cfg.artifacts_dir / "experiments" / "nested_texture_kernel"
    paths = {
        "features": write_csv(photos, output / "photo_features.csv"),
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(pd.concat(all_views, ignore_index=True), output / "camera_predictions.csv"),
        "outer_selection": write_csv(outer_selection, output / "outer_selection.csv"),
        "final_selection": write_csv(final_selection, output / "final_selection.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / "rgb_texture_nested_kernel.csv"),
    }
    sources = [Path(config_path), cfg.curated_file("train"), cfg.curated_file("sample_submission"), index_path,
               rgb_path, texture_path, manifest_path, *reference_paths.values()]
    paths["summary"] = write_json({
        "kernel": "rbf", "feature_count": 17, "crop_mm": 100, "aggregation": "Equal float64 photo means per soil",
        "grid": [{"alpha": a, "gamma_scale": g, "gamma": g / 17} for a, g in KERNEL_SETTINGS],
        "fixed_diagnostic": {"alpha": 1., "gamma": 1. / 17},
        "final_parameters": {"alpha": alpha, "gamma_scale": gamma_scale, "gamma": gamma_scale / 17},
        "validation": "Nested whole-soil LOO; train-only scaling/kernel centering; one shared parameter pair across targets and camera views",
        "final_selection_note": "Full-training selection EMD chooses final parameters; it is not a validation estimate",
        "tie_rule": "Larger alpha, then smaller gamma", "train_samples": len(train_ids), "test_samples": len(test_ids),
        "incumbent_emd": float(reference_errors.mean()), "incumbent_camera_disagreement_emd": reference_camera,
        "reference_max_abs_differences": differences, "experiments": summaries, "submission_evidence": evidence,
        "submission_rule": "Only nested candidate; at most one if >=1 EMD gain, >=half soils improve, camera not worse, positive gain without largest beneficiary",
        "selected_submission": str(paths["submission"]) if evidence["eligible"] else None,
        "submission": {"path": str(paths["submission"]), "sha256": sha256(paths["submission"].read_bytes()).hexdigest()},
        "feature_cache": {"path": str(paths["features"]), "sha256": sha256(paths["features"].read_bytes()).hexdigest()},
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
    }, output / "summary.json")
    print(f"Nested candidate submission screen: {evidence['eligible']}", flush=True)
    return paths
