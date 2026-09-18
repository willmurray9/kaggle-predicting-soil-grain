from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.official_experiment import _aligned_predictions
from soilgrain.pls import pls_curves
from soilgrain.ridge import ridge_curves
from soilgrain.search_inputs import _verified_sources, load_search_inputs
from soilgrain.targets import curve_array, validate_cumulative_curves
from soilgrain.texture_experiment import _aligned_cache


CAMERAS = ("Motorola Edge", "Samsung A52")
# Saved September 16 pooled PLS results; every outer fold selected one component.
PINNED_POOLED_DINO = {
    "experiments/autonomous_search/dino_pls/oof.csv": "b636f551a1a6846c5cdacf3d52df8bc82e9b38c4c44f372ada501a3dad3e38a0",
    "experiments/autonomous_search/dino_pls/outer_selection.csv": "57b3a20595acb5c2599cbbd083250230a074a2c170a0fac5b7b21b57f0be79cb",
}


def evaluate_camera_transfer(
    photos: pd.DataFrame, sample_ids: list[str], curves: np.ndarray, predictor: Callable,
) -> pd.DataFrame:
    """Hold out both views of each paired soil; match transfer/control training IDs."""
    ids, curves = pd.Index(sample_ids), np.asarray(curves, dtype=float)
    columns = [c for c in photos if isinstance(c, str) and c.startswith("feature_")]
    if ids.has_duplicates or ids.hasnans or curves.shape != (len(ids), 11):
        raise ValueError("Require unique complete soil IDs and one 11-value curve per soil")
    validate_cumulative_curves(pd.DataFrame(curves, columns=CANONICAL_GRAIN_LABELS))
    if (not photos.columns.is_unique or not columns
            or not {"split", "sample_id", "camera"} <= set(photos)
            or photos[["split", "sample_id", "camera"]].isna().any().any()):
        raise ValueError("Photos require complete split/soil/camera keys and feature columns")
    train = photos.loc[photos.split == "train"]
    if set(train.sample_id) != set(ids):
        raise ValueError("Training photo soil coverage differs from label IDs")
    train = train.loc[train.camera.isin(CAMERAS)]
    if not np.isfinite(train[columns].to_numpy(dtype=float)).all():
        raise ValueError("Training features must be finite")
    coverage = train.groupby("sample_id").camera.nunique()
    paired_ids = [soil for soil in sample_ids if coverage.get(soil, 0) == 2]
    if len(paired_ids) < 3:
        raise ValueError("Camera transfer requires at least three paired soils")
    curves = curves[ids.get_indexer(paired_ids)]
    means = train.groupby(["sample_id", "camera"])[columns].mean()
    counts = train.groupby(["sample_id", "camera"]).size()
    features = {camera: means.xs(camera, level="camera").loc[paired_ids].to_numpy()
                for camera in CAMERAS}
    widths = np.diff(np.log10(SUPPORT_DIAMETERS))
    rows = []
    for i, soil in enumerate(paired_ids):
        keep = np.arange(len(paired_ids)) != i
        training_ids = [value for value in paired_ids if value != soil]
        query = np.vstack([features[camera][i] for camera in CAMERAS])
        for train_camera in CAMERAS:
            predictions = np.asarray(predictor(features[train_camera][keep], curves[keep], query), dtype=float)
            if predictions.shape != (2, 11):
                raise ValueError("Predictor must return one 11-value curve per query camera")
            validate_cumulative_curves(pd.DataFrame(predictions, columns=CANONICAL_GRAIN_LABELS))
            for query_camera, prediction in zip(CAMERAS, predictions, strict=True):
                contributions = np.abs(curves[i, :10] - prediction[:10]) * widths
                rows.append({
                    "sample_id": soil, "train_camera": train_camera, "query_camera": query_camera,
                    "direction": f"{train_camera} -> {query_camera}",
                    "training_soil_ids": json.dumps(training_ids), "training_soils": len(training_ids),
                    "query_photos": int(counts.loc[(soil, query_camera)]),
                    "emd": emd_score(curves[i], prediction), "fine_emd": float(contributions[:4].sum()),
                    **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
                    **{f"emd_contribution_{label}": float(value) for label, value in
                       zip(CANONICAL_GRAIN_LABELS[:10], contributions, strict=True)},
                })
    return pd.DataFrame(rows)


def summarize_camera_transfer(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report equal-soil means and pair each transfer with its source-camera control."""
    keys = ["sample_id", "train_camera"]
    if predictions.duplicated(keys + ["query_camera"]).any():
        raise ValueError("Camera predictions require unique soil/train/query keys")
    controls = predictions.loc[predictions.train_camera == predictions.query_camera]
    transfers = predictions.loc[predictions.train_camera != predictions.query_camera]
    columns = keys + ["training_soil_ids", "emd", "fine_emd"]
    paired = transfers[columns + ["direction"]].merge(
        controls[columns], on=keys, how="outer", validate="one_to_one", suffixes=("_transfer", "_control"),
    )
    if (paired.isna().any().any() or len(paired) * 2 != len(predictions)
            or not paired.training_soil_ids_transfer.eq(paired.training_soil_ids_control).all()):
        raise ValueError("Transfer/control folds require identical training soil IDs and coverage")
    paired = paired.rename(columns={"emd_transfer": "transfer_emd", "emd_control": "control_emd",
                                    "fine_emd_transfer": "transfer_fine_emd", "fine_emd_control": "control_fine_emd",
                                    "training_soil_ids_transfer": "training_soil_ids"})
    paired = paired.drop(columns="training_soil_ids_control")
    paired["transfer_minus_control_emd"] = paired.transfer_emd - paired.control_emd
    paired["transfer_minus_control_fine_emd"] = paired.transfer_fine_emd - paired.control_fine_emd
    contributions = [f"emd_contribution_{label}" for label in CANONICAL_GRAIN_LABELS[:10]]
    directions = predictions.groupby(["direction", "train_camera", "query_camera"], sort=False).agg(
        macro_soil_emd=("emd", "mean"), macro_soil_fine_emd=("fine_emd", "mean"), soils=("sample_id", "nunique"),
        **{column: (column, "mean") for column in contributions},
    ).reset_index()
    return directions, paired


def run_camera_transfer(config_path: str | Path = "configs/data.yaml") -> dict:
    """Run the three frozen recipes on the original 21 paired camera soils."""
    cfg = load_config(config_path)
    truth, _sample, photos, components, sources = load_search_inputs(config_path)
    sample_ids, curves = truth.sample_id.tolist(), curve_array(truth)
    if len(sample_ids) != 24:
        raise ValueError("The declared camera diagnostic requires the original 24 labeled soils")
    first = cfg.artifacts_dir / "experiments/first_batch"
    rgb_path, rgb_oof_path = first / "reference_rgb_100_photo_features.csv", first / "oof_predictions.csv"
    verified = {Path(item["path"]).resolve() for item in sources}
    if not {rgb_path.resolve(), rgb_oof_path.resolve()} <= verified:
        raise ValueError("Original RGB cache and pooled reference must be verified sources")
    keys = ["split", "sample_id", "camera", "path"]
    rgb = _aligned_cache(rgb_path, photos["spectral"][keys], 13)
    rgb_columns = [f"feature_{i}" for i in range(13)]
    difference = float(np.max(np.abs(rgb[rgb_columns].to_numpy() - photos["spectral"][rgb_columns].to_numpy())))
    if difference > 1e-12:
        raise ValueError(f"RGB feature mapping differs from the original cache: {difference}")

    sources += _verified_sources([
        {"path": str(cfg.artifacts_dir / path), "sha256": digest} for path, digest in PINNED_POOLED_DINO.items()
    ])
    dino_dir = cfg.artifacts_dir / "experiments/autonomous_search/dino_pls"
    selected = pd.read_csv(dino_dir / "outer_selection.csv")
    selected = selected.loc[selected.selected.eq(True)]
    if (selected.sample_id.duplicated().any() or set(selected.sample_id) != set(sample_ids)
            or not selected.n_components.eq(1).all()):
        raise ValueError("Saved pooled DINO reference must use one component in all 24 held-out folds")
    rgb_oof = pd.read_csv(rgb_oof_path)
    saved = {
        "rgb13_ridge": rgb_oof.loc[rgb_oof.experiment == "ridge_rgb_100"],
        "spectral23_ridge": components["spectral"]["oof"],
        "dino_pls1": pd.read_csv(dino_dir / "oof.csv"),
    }
    pooled = []
    for name, frame in saved.items():
        aligned = _aligned_predictions(frame, truth, ["sample_id"])
        pooled.append({"experiment": name, "soils": len(sample_ids), "training_soils_per_fold": 23,
                       "macro_soil_emd": emd_score(curves, curve_array(aligned))})

    all_predictions, all_directions, all_paired, recipes = [], [], [], []
    for name, frame, predictor, settings in (
        ("rgb13_ridge", rgb, partial(ridge_curves, alpha=10.), {"feature_count": 13, "alpha": 10.}),
        ("spectral23_ridge", photos["spectral"], partial(ridge_curves, alpha=10.), {"feature_count": 23, "alpha": 10.}),
        ("dino_pls1", photos["dino"], partial(pls_curves, n_components=1), {"feature_count": 384, "n_components": 1}),
    ):
        predictions = evaluate_camera_transfer(frame, sample_ids, curves, predictor)
        if predictions.sample_id.nunique() != 21 or not predictions.training_soils.eq(20).all():
            raise ValueError("The declared camera diagnostic requires exactly 21 paired soils and 20 training soils")
        directions, paired = summarize_camera_transfer(predictions)
        for result in (predictions, directions, paired):
            result.insert(0, "experiment", name)
        all_predictions.append(predictions)
        all_directions.append(directions)
        all_paired.append(paired)
        recipes.append({"experiment": name, **settings})
    predictions = pd.concat(all_predictions, ignore_index=True)
    directions = pd.concat(all_directions, ignore_index=True)
    paired = pd.concat(all_paired, ignore_index=True)
    paired_summary = paired.groupby(["experiment", "direction"], sort=False).agg(
        macro_soil_transfer_minus_control_emd=("transfer_minus_control_emd", "mean"),
        median_transfer_minus_control_emd=("transfer_minus_control_emd", "median"),
        macro_soil_transfer_minus_control_fine_emd=("transfer_minus_control_fine_emd", "mean"),
        soils=("sample_id", "nunique"),
    ).reset_index()
    output = cfg.artifacts_dir / "experiments/camera_transfer"
    paths = {
        "predictions": write_csv(predictions, output / "predictions.csv"),
        "directions": write_csv(directions, output / "direction_summary.csv"),
        "paired_comparison": write_csv(paired, output / "per_soil_transfer_minus_control.csv"),
    }
    code_dir = Path(__file__).resolve().parent
    source_paths = [code_dir / name for name in ("camera_transfer.py", "ridge.py", "pls.py", "metrics.py", "search_inputs.py")]
    sources += [{"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()} for path in source_paths]
    paired_ids = predictions.sample_id.drop_duplicates().tolist()
    summary = {
        "producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=code_dir).strip(),
        "paired_soils": len(paired_ids), "paired_soil_ids": paired_ids,
        "excluded_soil_ids": [soil for soil in sample_ids if soil not in paired_ids],
        "training_soils_per_fold": 20, "cameras": list(CAMERAS), "recipes": recipes,
        "aggregation": "Mean features within each soil/camera; one equally weighted training row and score per soil",
        "validation": "Both views of held-out soil excluded; identical training IDs for each source-camera transfer and control; no tuning",
        "fine_emd_definition": "EMD contributions at 0.002, 0.0063, 0.02 and 0.063 mm (first four log-width intervals)",
        "rgb_feature_mapping": {column: column for column in rgb_columns},
        "rgb_mapping_max_abs_difference": difference,
        "directions": directions.to_dict(orient="records"),
        "paired_transfer_minus_control": paired_summary.to_dict(orient="records"),
        "existing_pooled_loo": pooled,
        "pooled_reference_note": "Saved 24-soil pooled-photo LOO has 23 training soils; its means are separate from the 21-soil camera folds. Saved DINO nested selection chose one component in every fold.",
        "limitation": "Transfer between the two original cameras does not establish performance on unseen iPhones or German soils",
        "sources": sources,
        "outputs": {name: {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()} for name, path in paths.items()},
    }
    write_json(summary, output / "summary.json")
    return summary
