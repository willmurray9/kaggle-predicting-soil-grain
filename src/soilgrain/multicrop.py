from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


COMPONENTS = ("gray_50", "gray_100", "gray_150")
EXPERIMENT = "multicrop_baseline"


def _row_keys(frame: pd.DataFrame, keys: Sequence[str]) -> list[tuple[object, ...]]:
    if not keys or any(key not in frame for key in keys):
        raise ValueError(f"Missing alignment keys: {list(keys)}")
    if frame[list(keys)].isna().any().any():
        raise ValueError(f"Alignment keys contain null values: {list(keys)}")
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"Component contains duplicate records for keys {list(keys)}")
    return list(frame[list(keys)].itertuples(index=False, name=None))


def blend_predictions(
    components: dict[str, pd.DataFrame], keys: Sequence[str]
) -> pd.DataFrame:
    """Average three crop predictions after exact key and curve validation."""
    if set(components) != set(COMPONENTS):
        raise ValueError(f"Expected exactly these components: {list(COMPONENTS)}")

    base_keys: list[tuple[object, ...]] | None = None
    aligned = []
    for name in COMPONENTS:
        frame = components[name]
        if frame.empty:
            raise ValueError(f"Component {name} is empty")
        validate_cumulative_curves(frame)
        row_keys = _row_keys(frame, keys)
        if base_keys is None:
            base_keys = row_keys
        elif set(row_keys) != set(base_keys):
            missing = set(base_keys) - set(row_keys)
            extra = set(row_keys) - set(base_keys)
            raise ValueError(
                f"Component {name} key coverage mismatch: "
                f"missing={list(missing)[:5]}, extra={list(extra)[:5]}"
            )
        positions = {key: i for i, key in enumerate(row_keys)}
        aligned.append(curve_array(frame.iloc[[positions[key] for key in base_keys]]))

    assert base_keys is not None
    output = pd.DataFrame(base_keys, columns=list(keys))
    output[list(CANONICAL_GRAIN_LABELS)] = np.mean(aligned, axis=0)
    validate_cumulative_curves(output)
    return output


def _components_from_experiment_file(path: Path, keys: Sequence[str]) -> dict[str, pd.DataFrame]:
    source = pd.read_csv(path)
    required = {"experiment", "emd", *keys}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    ordered_grain_columns(source)
    return {
        name: source.loc[source["experiment"] == name, [*keys, *ordered_grain_columns(source)]].copy()
        for name in COMPONENTS
    }


def _score_rows(truth: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    if truth["sample_id"].duplicated().any():
        raise ValueError("Training labels contain duplicate sample_id values")
    truth_by_id = truth.set_index("sample_id")
    predicted_ids = predictions["sample_id"].tolist()
    missing = set(predicted_ids) - set(truth_by_id.index)
    if missing:
        raise ValueError(f"Predictions lack training labels: {sorted(missing)[:5]}")
    actual = curve_array(truth_by_id.loc[predicted_ids].reset_index())
    predicted = curve_array(predictions)
    return np.array([emd_score(a, p) for a, p in zip(actual, predicted, strict=True)])


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_multicrop(
    config_path: str | Path = "configs/data.yaml",
) -> dict[str, Path]:
    """Write the fixed equal blend from the three first-batch grayscale crops."""
    cfg = load_config(config_path)
    first_batch = cfg.artifacts_dir / "experiments" / "first_batch"
    oof_source = first_batch / "oof_predictions.csv"
    camera_source = first_batch / "camera_predictions.csv"
    submission_sources = {
        name: cfg.submissions_dir / "experiments" / f"{name}.csv" for name in COMPONENTS
    }
    source_paths = [oof_source, camera_source, *submission_sources.values()]
    missing_paths = [str(path) for path in source_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing multicrop component files: {missing_paths}")

    truth = pd.read_csv(cfg.curated_file("train"))
    sample = pd.read_csv(cfg.curated_file("sample_submission"))
    validate_cumulative_curves(truth)

    oof = blend_predictions(
        _components_from_experiment_file(oof_source, ["sample_id"]), ["sample_id"]
    )
    expected_ids = set(truth["sample_id"])
    if set(oof["sample_id"]) != expected_ids:
        raise ValueError("OOF sample_id coverage does not match training labels")
    oof_errors = _score_rows(truth, oof)
    oof.insert(0, "experiment", EXPERIMENT)
    oof.insert(2, "emd", oof_errors)

    cameras = blend_predictions(
        _components_from_experiment_file(camera_source, ["sample_id", "camera"]),
        ["sample_id", "camera"],
    )
    camera_errors = _score_rows(truth, cameras)
    cameras.insert(0, "experiment", EXPERIMENT)
    cameras.insert(3, "emd", camera_errors)
    disagreements = [
        emd_score(a, b)
        for _, group in cameras.groupby("sample_id", sort=False)
        for a, b in combinations(curve_array(group), 2)
    ]

    candidates = {}
    for name, path in submission_sources.items():
        candidate = pd.read_csv(path)
        validate_submission(candidate, sample)
        candidates[name] = candidate
    submission_curves = blend_predictions(candidates, ["sample_id"])
    positions = submission_curves.set_index("sample_id")
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = curve_array(
        positions.loc[sample["sample_id"]].reset_index()
    )
    validate_submission(submission, sample)

    output = cfg.artifacts_dir / "experiments" / "multicrop"
    paths = {
        "oof": write_csv(oof, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(cameras, output / "camera_predictions.csv"),
        "submission": write_csv(
            submission, cfg.submissions_dir / "multicrop_baseline.csv"
        ),
    }
    paths["summary"] = write_json(
        {
            "experiment": EXPERIMENT,
            "components": list(COMPONENTS),
            "blend": "Equal arithmetic mean of the three component curves; no tuned weights",
            "loo_emd": float(oof_errors.mean()),
            "paired_camera_disagreement_emd": float(np.mean(disagreements)) if disagreements else None,
            "camera_pairs": len(disagreements),
            "train_samples": len(oof),
            "test_samples": len(submission),
            "validation": "Precomputed leave-one-physical-soil-out predictions aligned by sample_id",
            "component_sources": [
                {"path": str(path), "sha256": _sha256(path)} for path in source_paths
            ],
        },
        output / "summary.json",
    )
    return paths
