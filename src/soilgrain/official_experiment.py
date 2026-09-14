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
from soilgrain.model_blend import _blend_predictions
from soilgrain.multicrop import _score_rows
from soilgrain.nested_ridge import RIDGE_ALPHAS, evaluate_nested_ridge, select_ridge_alpha
from soilgrain.pca_experiment import _camera_disagreement, submission_evidence
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def recipe_submission_evidence(errors, incumbent_errors, camera, incumbent_camera, legacy_errors, legacy_camera):
    incumbent = submission_evidence(errors, incumbent_errors, camera, incumbent_camera)
    legacy = submission_evidence(errors, legacy_errors, camera, legacy_camera)
    return {
        **incumbent, "incumbent_eligible": incumbent["eligible"],
        "legacy_mean_improvement_emd": legacy["mean_improvement_emd"],
        "legacy_camera_not_worse": legacy["camera_not_worse"],
        "eligible": bool(incumbent["eligible"] and legacy["mean_improvement_emd"] >= 1.0 and legacy["camera_not_worse"]),
    }


def _aligned_predictions(frame, expected, keys):
    validate_cumulative_curves(frame)
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Reference predictions require complete, unique keys")
    frame, expected = frame.set_index(keys), expected.set_index(keys)
    if set(frame.index) != set(expected.index):
        raise ValueError("Reference prediction coverage differs from the experiment")
    return frame.loc[expected.index].reset_index()


def write_official_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    truth, sample, ppm = [pd.read_csv(cfg.curated_file(key)) for key in ("train", "sample_submission", "ppm")]
    validate_cumulative_curves(truth)
    for frame in (truth, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels/template require nonempty, unique, complete soil IDs")
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    if len(train_ids) < 11 or set(train_ids) & set(test_ids):
        raise ValueError("Eight-component nested PCA requires at least 11 training soils and disjoint test soils")
    keys, columns = ["split", "sample_id", "camera", "path"], [f"feature_{i}" for i in range(512)]
    index_path = cfg.reports_dir / "photo_index.csv"
    index = pd.read_csv(index_path)
    if index[keys].isna().any().any() or index["path"].duplicated().any() or set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index requires unique paths, complete keys, and train/test splits")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(ids):
            raise ValueError(f"Photo index {split} coverage differs from labels/template")
    if ppm["phone"].isna().any() or ppm["phone"].duplicated().any():
        raise ValueError("Camera calibration requires complete, unique phones")
    frozen = cfg.artifacts_dir / "experiments" / "frozen_resnet18"
    encoder_path, cache_path = frozen / "summary.json", frozen / "photo_features.csv"
    old_index = pd.read_csv(cache_path, usecols=keys)
    if len(old_index) != len(index) or set(pd.MultiIndex.from_frame(old_index[keys])) != set(pd.MultiIndex.from_frame(index[keys])):
        raise ValueError("Legacy feature cache coverage differs from the photo index")
    old_manifest = json.loads(encoder_path.read_text())
    for item in old_manifest["sources"]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Legacy encoder source changed: {item['path']}")
    photo_sources = [{"path": p, "sha256": sha256(Path(p).read_bytes()).hexdigest()} for p in index["path"]]
    if {p["path"]: p["sha256"] for p in photo_sources} != {p["path"]: p["sha256"] for p in old_manifest["photo_sources"]}:
        raise ValueError("Source photos differ from the legacy encoder experiment")
    sources = [Path(config_path), *(cfg.curated_file(k) for k in ("train", "sample_submission", "ppm")),
               index_path, encoder_path, cache_path]
    expected_views = index.loc[index["split"] == "train", ["sample_id", "camera"]].drop_duplicates()
    references = {}
    for name, directory, experiment in (
        ("incumbent", "physical_texture", "ridge_rgb_texture_100"),
        ("legacy_pca", "pca_ridge", "pca"), ("legacy_blend", "pca_ridge", "blend"),
    ):
        folder = cfg.artifacts_dir / "experiments" / directory
        reference = {}
        for kind, expected, prediction_keys in (("oof", truth, ["sample_id"]), ("camera", expected_views, ["sample_id", "camera"])):
            path = folder / f"{kind}_predictions.csv"
            frame = pd.read_csv(path)
            reference[kind] = _aligned_predictions(frame.loc[frame["experiment"] == experiment], expected, prediction_keys)
            sources.append(path)
        reference["errors"] = _score_rows(truth, reference["oof"])
        reference["camera_emd"], reference["camera_pairs"] = _camera_disagreement(reference["camera"])
        references[name] = reference
        manifest_path = folder / "summary.json"
        sources.append(manifest_path)
        for item in json.loads(manifest_path.read_text()).get("sources", []):
            if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Saved reference source changed: {item['path']}")
    incumbent_path = cfg.submissions_dir / "ridge_rgb_texture_100.csv"
    incumbent_submission = pd.read_csv(incumbent_path)
    validate_submission(incumbent_submission, sample)
    sources.append(incumbent_path)

    photos, encoder = extract_frozen_features(index, ppm, preprocessing="official")
    if encoder["checkpoint_sha256"] != old_manifest["encoder"]["checkpoint_sha256"]:
        raise ValueError("Official and legacy preprocessing require the same checkpoint")
    if (not photos[keys].reset_index(drop=True).equals(index[keys].reset_index(drop=True))
            or not np.isfinite(photos[columns].to_numpy()).all()):
        raise ValueError("Extracted features must be finite and preserve photo keys")
    photos = photos.astype({c: np.float32 for c in columns})
    train_photos = photos.loc[photos["split"] == "train"]
    train_features = train_photos.groupby("sample_id")[columns].mean().loc[train_ids].to_numpy()
    test_features = photos.loc[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy()
    curves = curve_array(truth)
    oof, views, outer_selection = evaluate_nested_ridge(train_photos, train_ids, curves, n_components=8)
    pca = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    pca.insert(0, "sample_id", train_ids)
    alpha, scores = select_ridge_alpha(train_features, curves, n_components=8)
    final_selection = pd.DataFrame([{"alpha": a, "loo_selection_emd": s, "selected": a == alpha} for a, s in scores.items()])
    pca_submission = sample.copy()
    pca_submission[ordered_grain_columns(sample)] = ridge_curves(train_features, curves, test_features, alpha=alpha, n_components=8)
    validate_submission(pca_submission, sample)
    incumbent = references["incumbent"]
    blend = _blend_predictions(pca, incumbent["oof"], ["sample_id"])
    blend_views = _blend_predictions(views, incumbent["camera"], ["sample_id", "camera"])
    blend_test = _blend_predictions(pca_submission, incumbent_submission, ["sample_id"])
    blend_submission = sample.copy()
    blend_submission[ordered_grain_columns(sample)] = curve_array(blend_test.set_index("sample_id").loc[test_ids])
    validate_submission(blend_submission, sample)
    comparison = pd.DataFrame({"sample_id": train_ids, **{f"emd_{name}": ref["errors"] for name, ref in references.items()}})
    summaries, all_oof, all_views = [], [], []
    for name, predicted, camera_predictions in (("pca", pca, views), ("blend", blend, blend_views)):
        predicted["emd"] = _score_rows(truth, predicted)
        camera_predictions["emd"] = _score_rows(truth, camera_predictions)
        errors = predicted["emd"].to_numpy()
        camera_emd, pairs = _camera_disagreement(camera_predictions)
        legacy = references[f"legacy_{name}"]
        summaries.append({"experiment": name, "outer_loo_emd": float(errors.mean()),
                          "paired_camera_disagreement_emd": camera_emd, "camera_pairs": pairs,
                          **recipe_submission_evidence(errors, incumbent["errors"], camera_emd, incumbent["camera_emd"], legacy["errors"], legacy["camera_emd"])})
        comparison[f"emd_{name}"] = errors
        comparison[f"improvement_{name}_vs_incumbent"] = incumbent["errors"] - errors
        comparison[f"improvement_{name}_vs_legacy"] = legacy["errors"] - errors
        predicted.insert(0, "experiment", name)
        camera_predictions.insert(0, "experiment", name)
        all_oof.append(predicted)
        all_views.append(camera_predictions)
        print(f"Official {name}: {errors.mean():.4f} EMD; submission screen={summaries[-1]['eligible']}", flush=True)
    output = cfg.artifacts_dir / "experiments" / "official_preprocessing"
    paths = {
        "features": write_csv(photos, output / "photo_features.csv"),
        "oof": write_csv(pd.concat(all_oof, ignore_index=True), output / "oof_predictions.csv"),
        "camera_predictions": write_csv(pd.concat(all_views, ignore_index=True), output / "camera_predictions.csv"),
        "outer_selection": write_csv(outer_selection, output / "outer_selection.csv"),
        "final_selection": write_csv(final_selection, output / "final_selection.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "pca_submission": write_csv(pca_submission, cfg.submissions_dir / "official_resnet18_pca8_nested_ridge.csv"),
        "blend_submission": write_csv(blend_submission, cfg.submissions_dir / "rgb_texture_official_pca8_blend.csv"),
    }
    eligible = [row for row in summaries if row["eligible"]]
    selected = min(eligible, key=lambda row: row["outer_loo_emd"])["experiment"] if eligible else None
    paths["summary"] = write_json({
        "encoder": encoder, "input_features": 512, "pca_components": 8,
        "pca": "Training-only standardization and SVD projection; no whitening or component rescaling",
        "ridge_alphas": RIDGE_ALPHAS, "final_alpha": alpha, "blend_weights": {"pca": 0.5, "rgb_texture": 0.5},
        "validation": "Nested whole-soil LOO; refit PCA, scaling, and ridge in every inner/outer fit",
        "final_selection_note": "Full-training scores choose final alpha; they are not validation estimates",
        "aggregation": "Equal photo means per soil, preserving float32 aggregation",
        "train_samples": len(train_ids), "test_samples": len(test_ids), "experiments": summaries,
        "references": {name: {"outer_loo_emd": float(ref["errors"].mean()), "paired_camera_disagreement_emd": ref["camera_emd"],
                              "camera_pairs": ref["camera_pairs"]} for name, ref in references.items()},
        "submission_rule": "All four incumbent screens plus >= 1 EMD gain and no worse camera disagreement versus matched legacy candidate; at most one, lowest EMD, tie PCA",
        "selected_submission": str(paths[f"{selected}_submission"]) if selected else None,
        "submissions": [{"path": str(paths[key]), "sha256": sha256(paths[key].read_bytes()).hexdigest()} for key in ("pca_submission", "blend_submission")],
        "feature_cache": {"path": str(paths["features"]), "sha256": sha256(paths["features"].read_bytes()).hexdigest()},
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(sources)],
        "photo_sources": photo_sources,
    }, output / "summary.json")
    return paths
