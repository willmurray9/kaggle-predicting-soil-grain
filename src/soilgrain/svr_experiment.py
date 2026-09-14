from __future__ import annotations

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.io import write_csv, write_json
from soilgrain.multicrop import _score_rows
from soilgrain.nested_svr import SVR_C_VALUES, evaluate_nested_svr, select_svr_c
from soilgrain.pca_experiment import _camera_disagreement, submission_evidence
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.svr import SVR_EPSILON, SVR_MAX_ITER, SVR_TOL, svr_curves
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.texture_experiment import _aligned_cache, _reference_difference


def write_svr_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
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
    cache_dir = cfg.artifacts_dir / "experiments" / "nested_texture_kernel"
    cache_path, manifest_path = cache_dir / "photo_features.csv", cache_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["feature_cache"]["path"]).resolve() != cache_path.resolve():
        raise ValueError("Feature-cache source changed from the recorded experiment")
    for item in [*manifest["sources"], manifest["feature_cache"]]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Recorded source changed: {item['path']}")
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

    oof, views, outer_selection = evaluate_nested_svr(train_photos, train_ids, curves)
    predictions = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    predictions.insert(0, "sample_id", train_ids)
    predictions["emd"] = _score_rows(truth, predictions)
    views["emd"] = _score_rows(truth, views)
    errors = predictions["emd"].to_numpy()
    camera_emd, pairs = _camera_disagreement(views)
    evidence = submission_evidence(errors, reference_errors, camera_emd, reference_camera)
    print(f"Nested linear SVR: {errors.mean():.4f} EMD; camera disagreement {camera_emd}; submission screen={evidence['eligible']}", flush=True)
    C, scores = select_svr_c(train_features, curves)
    final_selection = pd.DataFrame([{"C": c, "loo_selection_emd": score, "selected": c == C} for c, score in scores.items()])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = svr_curves(train_features, curves, test_features, C=C)
    validate_submission(submission, sample)
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_incumbent": reference_errors, "emd_svr": errors,
                               "improvement_vs_incumbent": reference_errors - errors})
    output = cfg.artifacts_dir / "experiments" / "linear_svr"
    paths = {
        "oof": write_csv(predictions, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(views, output / "camera_predictions.csv"),
        "outer_selection": write_csv(outer_selection, output / "outer_selection.csv"),
        "final_selection": write_csv(final_selection, output / "final_selection.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / "rgb_texture_linear_svr.csv"),
    }
    sources = [Path(config_path), cfg.curated_file("train"), cfg.curated_file("sample_submission"), index_path,
               cache_path, manifest_path, *reference_paths.values(), Path("pyproject.toml"), Path("uv.lock")]
    paths["summary"] = write_json({
        "model": "sklearn.svm.SVR", "kernel": "linear", "feature_count": 17, "crop_mm": 100,
        "aggregation": "Equal float64 photo means per soil", "C_grid": SVR_C_VALUES, "final_C": C,
        "epsilon": SVR_EPSILON, "tol": SVR_TOL, "max_iter": SVR_MAX_ITER, "shrinking": True,
        "solver_checks": "Every fit requires successful status; convergence warnings and nonfinite predictions are fatal",
        "targets": "Ten independent raw-percentage regressions sharing C; clip/monotone repair; final endpoint 100",
        "validation": "Nested whole-soil LOO; train-only feature scaling; one shared C for pooled and camera queries",
        "final_selection_note": "Full-training selection EMD chooses final C; it is not a validation estimate",
        "tie_rule": "Smaller C", "train_samples": len(train_ids), "test_samples": len(test_ids),
        "outer_loo_emd": float(errors.mean()), "paired_camera_disagreement_emd": camera_emd, "camera_pairs": pairs,
        "incumbent_emd": float(reference_errors.mean()), "incumbent_camera_disagreement_emd": reference_camera,
        "reference_max_abs_differences": differences, "submission_evidence": evidence,
        "submission_rule": "At most one if >=1 EMD gain, >=half soils improve, camera not worse, positive gain without largest beneficiary",
        "selected_submission": str(paths["submission"]) if evidence["eligible"] else None,
        "submission": {"path": str(paths["submission"]), "sha256": sha256(paths["submission"].read_bytes()).hexdigest()},
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
        "versions": {"python": python_version(), **{name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}},
    }, output / "summary.json")
    return paths
