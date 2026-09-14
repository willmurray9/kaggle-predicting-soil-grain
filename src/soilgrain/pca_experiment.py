from __future__ import annotations

import json
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.model_blend import _blend_predictions
from soilgrain.multicrop import _score_rows
from soilgrain.nested_ridge import RIDGE_ALPHAS, evaluate_nested_ridge, select_ridge_alpha
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.texture_experiment import _reference_difference


def submission_evidence(errors, reference_errors, camera_emd, reference_camera_emd) -> dict:
    """Apply the declared screen without changing fitting or the primary score."""
    errors, reference_errors = np.asarray(errors, dtype=float), np.asarray(reference_errors, dtype=float)
    if errors.shape != reference_errors.shape:
        raise ValueError("Submission evidence requires finite per-soil error pairs")
    improvements = reference_errors - errors
    if improvements.ndim != 1 or len(improvements) < 2 or not np.isfinite(improvements).all():
        raise ValueError("Submission evidence requires finite per-soil error pairs")
    gain = float(improvements.mean())
    count = int((improvements > 1e-9).sum())
    without_largest = float((improvements.sum() - improvements.max()) / (len(improvements) - 1))
    camera_not_worse = bool(camera_emd is not None and reference_camera_emd is not None
                            and np.isfinite(camera_emd) and np.isfinite(reference_camera_emd)
                            and camera_emd <= reference_camera_emd)
    return {
        "mean_improvement_emd": gain, "improved_soils": count,
        "improvement_without_largest_emd": without_largest, "camera_not_worse": camera_not_worse,
        "eligible": bool(gain >= 1.0 and count * 2 >= len(improvements) and camera_not_worse and without_largest > 0),
    }


def _check_selection(actual, saved, keys, score_column):
    for frame in (actual, saved):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Nested reference selection keys must be unique and complete")
    actual, saved = actual.set_index(keys), saved.set_index(keys)
    if set(actual.index) != set(saved.index):
        raise ValueError("Nested reference selection coverage differs")
    saved = saved.loc[actual.index]
    if (not np.allclose(actual[score_column], saved[score_column], rtol=0, atol=1e-10)
            or not np.array_equal(actual["selected"], saved["selected"])):
        raise ValueError("Nested reference alpha selections do not reconstruct")


def _camera_disagreement(views):
    errors = [emd_score(a, b) for _, group in views.groupby("sample_id") for a, b in combinations(curve_array(group), 2)]
    return (float(np.mean(errors)) if errors else None), len(errors)


def write_pca_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    truth, sample = pd.read_csv(cfg.curated_file("train")), pd.read_csv(cfg.curated_file("sample_submission"))
    validate_cumulative_curves(truth)
    for frame in (truth, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels/template require nonempty, unique, complete soil IDs")
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if len(train_ids) < 11 or set(train_ids) & set(test_ids):
        raise ValueError("Eight-component nested PCA requires at least 11 training soils and disjoint test soils")
    frozen = cfg.artifacts_dir / "experiments" / "frozen_resnet18"
    nested = cfg.artifacts_dir / "experiments" / "nested_ridge"
    incumbent_dir = cfg.artifacts_dir / "experiments" / "physical_texture"
    cache_path, encoder_path = frozen / "photo_features.csv", frozen / "summary.json"
    columns = [f"feature_{i}" for i in range(512)]
    photos = pd.read_csv(cache_path, dtype={column: np.float32 for column in columns})
    keys = ["split", "sample_id", "camera", "path"]
    if (set(c for c in photos if c.startswith("feature_")) != set(columns)
            or photos[keys].isna().any().any() or photos["path"].duplicated().any()
            or set(photos["split"]) != {"train", "test"} or not np.isfinite(photos[columns].to_numpy()).all()):
        raise ValueError("Frozen cache requires 512 finite features and unique, complete photo keys")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(photos.loc[photos["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Frozen {split} soil coverage differs from labels/template")
    train_photos = photos.loc[photos["split"] == "train", [*keys, *columns]]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    curves = curve_array(truth)
    sources = [Path(config_path), cfg.curated_file("train"), cfg.curated_file("sample_submission"), cache_path, encoder_path]
    reference_paths = {"oof": nested / "oof_predictions.csv", "camera": nested / "camera_predictions.csv",
                       "outer": nested / "outer_selection.csv", "final": nested / "final_selection.csv",
                       "submission": cfg.submissions_dir / "frozen_resnet18_nested_ridge.csv"}
    sources.extend([*reference_paths.values(), nested / "summary.json"])
    saved = {name: pd.read_csv(path) for name, path in reference_paths.items()}
    validate_submission(saved["submission"], sample)

    # Reconstruct the unchanged predictor before evaluating either new candidate.
    reference_oof, reference_views, reference_selection = evaluate_nested_ridge(train_photos, train_ids, curves)
    reference = pd.DataFrame(reference_oof, columns=CANONICAL_GRAIN_LABELS)
    reference.insert(0, "sample_id", train_ids)
    reference_alpha, reference_scores = select_ridge_alpha(train_features, curves)
    reference_submission = sample.copy()
    reference_submission[ordered_grain_columns(sample)] = ridge_curves(train_features, curves, test_features, alpha=reference_alpha)
    differences = {
        "oof": _reference_difference(reference, saved["oof"], ["sample_id"]),
        "camera": _reference_difference(reference_views, saved["camera"], ["sample_id", "camera"]),
        "submission": _reference_difference(reference_submission, saved["submission"], ["sample_id"]),
    }
    _check_selection(reference_selection, saved["outer"], ["sample_id", "alpha"], "inner_loo_emd")
    reference_final = pd.DataFrame([{"alpha": a, "loo_selection_emd": s, "selected": a == reference_alpha} for a, s in reference_scores.items()])
    _check_selection(reference_final, saved["final"], ["alpha"], "loo_selection_emd")
    print(f"Nested 512-feature reference reconstructed: {emd_score(curves, reference_oof):.4f} EMD", flush=True)

    incumbent_paths = {"oof": incumbent_dir / "oof_predictions.csv", "camera": incumbent_dir / "camera_predictions.csv",
                       "submission": cfg.submissions_dir / "ridge_rgb_texture_100.csv"}
    sources.extend([*incumbent_paths.values(), incumbent_dir / "summary.json"])
    incumbent = {name: pd.read_csv(path) for name, path in incumbent_paths.items()}
    for name in ("oof", "camera"):
        incumbent[name] = incumbent[name].loc[incumbent[name]["experiment"] == "ridge_rgb_texture_100"].copy()
    validate_submission(incumbent["submission"], sample)

    oof, views, outer_selection = evaluate_nested_ridge(train_photos, train_ids, curves, n_components=8)
    pca = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    pca.insert(0, "sample_id", train_ids)
    alpha, scores = select_ridge_alpha(train_features, curves, n_components=8)
    final_selection = pd.DataFrame([{"alpha": a, "loo_selection_emd": s, "selected": a == alpha} for a, s in scores.items()])
    pca_submission = sample.copy()
    pca_submission[ordered_grain_columns(sample)] = ridge_curves(train_features, curves, test_features, alpha=alpha, n_components=8)
    validate_submission(pca_submission, sample)
    blend = _blend_predictions(pca, incumbent["oof"], ["sample_id"])
    blend_views = _blend_predictions(views, incumbent["camera"], ["sample_id", "camera"])
    blend_test = _blend_predictions(pca_submission, incumbent["submission"], ["sample_id"])
    blend_submission = sample.copy()
    blend_submission[ordered_grain_columns(sample)] = curve_array(blend_test.set_index("sample_id").loc[test_ids])
    validate_submission(blend_submission, sample)
    incumbent_oof = incumbent["oof"].set_index("sample_id").loc[train_ids].reset_index()
    incumbent_errors = _score_rows(truth, incumbent_oof)
    incumbent_camera, camera_pairs = _camera_disagreement(incumbent["camera"])
    comparison = pd.DataFrame({"sample_id": train_ids, "emd_nested_reference": _score_rows(truth, reference), "emd_incumbent": incumbent_errors})
    summaries, all_oof, all_views = [], [], []
    for name, predicted, camera_predictions in (("pca", pca, views), ("blend", blend, blend_views)):
        predicted["emd"] = _score_rows(truth, predicted)
        camera_predictions["emd"] = _score_rows(truth, camera_predictions)
        camera_emd, pairs = _camera_disagreement(camera_predictions)
        errors = predicted["emd"].to_numpy()
        summaries.append({"experiment": name, "outer_loo_emd": float(errors.mean()),
                          "paired_camera_disagreement_emd": camera_emd, "camera_pairs": pairs,
                          **submission_evidence(errors, incumbent_errors, camera_emd, incumbent_camera)})
        comparison[f"emd_{name}"] = errors
        comparison[f"improvement_{name}_vs_incumbent"] = incumbent_errors - errors
        predicted.insert(0, "experiment", name)
        camera_predictions.insert(0, "experiment", name)
        all_oof.append(predicted)
        all_views.append(camera_predictions)
        print(f"{name}: {errors.mean():.4f} EMD; submission screen={summaries[-1]['eligible']}", flush=True)
    comparison["improvement_pca_vs_nested"] = comparison["emd_nested_reference"] - comparison["emd_pca"]
    output = cfg.artifacts_dir / "experiments" / "pca_ridge"
    paths = {
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(pd.concat(all_views, ignore_index=True), output / "camera_predictions.csv"),
        "outer_selection": write_csv(outer_selection, output / "outer_selection.csv"),
        "final_selection": write_csv(final_selection, output / "final_selection.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "pca_submission": write_csv(pca_submission, cfg.submissions_dir / "frozen_resnet18_pca8_nested_ridge.csv"),
        "blend_submission": write_csv(blend_submission, cfg.submissions_dir / "rgb_texture_pca8_blend.csv"),
    }
    eligible = [row for row in summaries if row["eligible"]]
    selected = min(eligible, key=lambda row: row["outer_loo_emd"])["experiment"] if eligible else None
    paths["summary"] = write_json({
        "encoder": json.loads(encoder_path.read_text())["encoder"], "input_features": 512, "pca_components": 8,
        "pca": "Training-only standardization and SVD projection; no whitening or component rescaling",
        "ridge_alphas": RIDGE_ALPHAS, "final_alpha": alpha, "blend_weights": {"pca": 0.5, "rgb_texture": 0.5},
        "validation": "Nested whole-soil LOO; refit PCA, scaling, and ridge in every inner/outer fit",
        "final_selection_note": "Full-training selection scores choose final alpha; they are not validation estimates",
        "aggregation": "Equal photo means per soil, preserving float32 cached aggregation",
        "train_samples": len(train_ids), "test_samples": len(test_ids), "camera_pairs": camera_pairs,
        "reference_max_abs_differences": differences, "reference_alpha_selections_match": True,
        "nested_reference_emd": emd_score(curves, reference_oof), "incumbent_emd": float(incumbent_errors.mean()),
        "incumbent_camera_disagreement_emd": incumbent_camera, "experiments": summaries,
        "submission_rule": "At most one: gain >= 1 EMD, >= half soils improve, no worse camera disagreement, positive gain without largest beneficiary; lowest EMD, tie PCA",
        "selected_submission": str(paths[f"{selected}_submission"]) if selected else None,
        "submissions": [{"path": str(paths[key]), "sha256": sha256(paths[key].read_bytes()).hexdigest()} for key in ("pca_submission", "blend_submission")],
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
    }, output / "summary.json")
    return paths
