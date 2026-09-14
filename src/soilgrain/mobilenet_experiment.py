from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.frozen_features import extract_frozen_features
from soilgrain.io import write_csv, write_json
from soilgrain.multicrop import _score_rows
from soilgrain.nested_ridge import RIDGE_ALPHAS, evaluate_nested_ridge, select_ridge_alpha
from soilgrain.official_experiment import _aligned_predictions
from soilgrain.pca_experiment import _camera_disagreement, submission_evidence
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def write_mobilenet_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    truth, sample, ppm = [pd.read_csv(cfg.curated_file(key)) for key in ("train", "sample_submission", "ppm")]
    validate_cumulative_curves(truth)
    for frame in (truth, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels/template require nonempty, unique, complete soil IDs")
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if len(train_ids) < 11 or set(train_ids) & set(test_ids):
        raise ValueError("Eight-component nested PCA requires at least 11 training soils and disjoint test soils")
    keys, columns = ["split", "sample_id", "camera", "path"], [f"feature_{i}" for i in range(960)]
    index_path = cfg.reports_dir / "photo_index.csv"
    index = pd.read_csv(index_path)
    if index[keys].isna().any().any() or index["path"].duplicated().any() or set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index requires unique paths, complete keys, and train/test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Photo index {split} coverage differs from labels/template")
    if ppm["phone"].isna().any() or ppm["phone"].duplicated().any():
        raise ValueError("Camera calibration requires complete, unique phones")
    official = cfg.artifacts_dir / "experiments" / "official_preprocessing"
    manifest_path = official / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    for item in [*manifest["sources"], manifest["feature_cache"], *manifest["submissions"]]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Recorded source changed: {item['path']}")
    photo_sources = [{"path": p, "sha256": sha256(Path(p).read_bytes()).hexdigest()} for p in index["path"]]
    if {p["path"]: p["sha256"] for p in photo_sources} != {p["path"]: p["sha256"] for p in manifest["photo_sources"]}:
        raise ValueError("Source photos differ from the official ResNet experiment")
    fallback_path = cfg.submissions_dir / "rgb_texture_official_pca8_blend.csv"
    if fallback_path.resolve() not in {Path(item["path"]).resolve() for item in manifest["submissions"]}:
        raise ValueError("Fallback candidate is missing from recorded sources")
    validate_submission(pd.read_csv(fallback_path), sample)
    sources = [Path(config_path), *(cfg.curated_file(k) for k in ("train", "sample_submission", "ppm")),
               index_path, manifest_path, fallback_path, Path("pyproject.toml"), Path("uv.lock")]
    expected_views = index.loc[index["split"] == "train", ["sample_id", "camera"]].drop_duplicates()
    references = {}
    for name, directory, experiment in (
        ("incumbent", "physical_texture", "ridge_rgb_texture_100"),
        ("official_pca", "official_preprocessing", "pca"),
        ("fallback_blend", "official_preprocessing", "blend"),
    ):
        reference = {}
        for kind, expected, prediction_keys in (("oof", truth, ["sample_id"]), ("camera", expected_views, ["sample_id", "camera"])):
            path = cfg.artifacts_dir / "experiments" / directory / f"{kind}_predictions.csv"
            frame = pd.read_csv(path)
            reference[kind] = _aligned_predictions(frame.loc[frame["experiment"] == experiment], expected, prediction_keys)
            sources.append(path)
        reference["errors"] = _score_rows(truth, reference["oof"])
        reference["camera_emd"], reference["camera_pairs"] = _camera_disagreement(reference["camera"])
        recorded = (manifest["references"]["incumbent"] if name == "incumbent" else
                    next(row for row in manifest["experiments"] if row["experiment"] == experiment))
        if not np.allclose([reference["errors"].mean(), reference["camera_emd"]],
                           [recorded["outer_loo_emd"], recorded["paired_camera_disagreement_emd"]], rtol=0, atol=1e-10):
            raise ValueError(f"Saved reference scores changed: {name}")
        references[name] = reference

    photos, encoder = extract_frozen_features(index, ppm, preprocessing="official", encoder_name="mobilenet_v3_large")
    if (not photos[keys].reset_index(drop=True).equals(index[keys].reset_index(drop=True))
            or set(c for c in photos if c.startswith("feature_")) != set(columns)
            or not np.isfinite(photos[columns].to_numpy()).all()):
        raise ValueError("Extracted features must be finite and preserve photo keys")
    photos = photos.astype({c: np.float32 for c in columns})
    train_photos = photos.loc[photos["split"] == "train"]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos.loc[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    curves = curve_array(truth)
    print("Source checks and feature extraction complete; evaluating MobileNet PCA ridge.", flush=True)
    oof, views, outer_selection = evaluate_nested_ridge(train_photos, train_ids, curves, n_components=8)
    predictions = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    predictions.insert(0, "sample_id", train_ids)
    predictions["emd"] = _score_rows(truth, predictions)
    views["emd"] = _score_rows(truth, views)
    errors = predictions["emd"].to_numpy()
    camera_emd, pairs = _camera_disagreement(views)
    incumbent = references["incumbent"]
    evidence = submission_evidence(errors, incumbent["errors"], camera_emd, incumbent["camera_emd"])
    alpha, scores = select_ridge_alpha(train_features, curves, n_components=8)
    final_selection = pd.DataFrame([{"alpha": a, "loo_selection_emd": s, "selected": a == alpha} for a, s in scores.items()])
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = ridge_curves(train_features, curves, test_features, alpha=alpha, n_components=8)
    validate_submission(submission, sample)
    comparison = pd.DataFrame({"sample_id": train_ids, **{f"emd_{name}": ref["errors"] for name, ref in references.items()},
                               "emd_mobilenet": errors, "improvement_vs_incumbent": incumbent["errors"] - errors})
    output = cfg.artifacts_dir / "experiments" / "mobilenet_pca"
    paths = {
        "features": write_csv(photos, output / "photo_features.csv"),
        "oof": write_csv(predictions, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(views, output / "camera_predictions.csv"),
        "outer_selection": write_csv(outer_selection, output / "outer_selection.csv"),
        "final_selection": write_csv(final_selection, output / "final_selection.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "submission": write_csv(submission, cfg.submissions_dir / "mobilenet_v3_large_pca8_nested_ridge.csv"),
    }
    selected = paths["submission"] if errors.mean() < references["fallback_blend"]["errors"].mean() else fallback_path
    print(f"MobileNet: {errors.mean():.4f} EMD; camera disagreement {camera_emd}; selected upload: {selected.name}", flush=True)
    paths["summary"] = write_json({
        "encoder": encoder, "input_features": 960, "pca_components": 8, "ridge_alphas": RIDGE_ALPHAS, "final_alpha": alpha,
        "pca": "Training-only standardization and SVD projection; no whitening or component rescaling",
        "validation": "Nested whole-soil LOO; refit scaling, PCA, and ridge in every inner/outer fit",
        "final_selection_note": "Full-training scores choose final alpha; they are not validation estimates",
        "aggregation": "Equal photo means per soil, preserving float32 aggregation",
        "train_samples": len(train_ids), "test_samples": len(test_ids), "outer_loo_emd": float(errors.mean()),
        "paired_camera_disagreement_emd": camera_emd, "camera_pairs": pairs, "incumbent_screen_diagnostic": evidence,
        "references": {name: {"outer_loo_emd": float(ref["errors"].mean()), "paired_camera_disagreement_emd": ref["camera_emd"],
                              "camera_pairs": ref["camera_pairs"]} for name, ref in references.items()},
        "submission_rule": "One informative upload requested: lower outer EMD of MobileNet and saved official blend; exact tie saved blend; old screens diagnostic only",
        "selected_submission": str(selected),
        "submission": {"path": str(paths["submission"]), "sha256": sha256(paths["submission"].read_bytes()).hexdigest()},
        "feature_cache": {"path": str(paths["features"]), "sha256": sha256(paths["features"].read_bytes()).hexdigest()},
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(sources)],
        "photo_sources": photo_sources,
    }, output / "summary.json")
    return paths
