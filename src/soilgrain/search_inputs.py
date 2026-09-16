from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.official_experiment import _aligned_predictions
from soilgrain.submission import validate_submission
from soilgrain.targets import validate_cumulative_curves
from soilgrain.texture_experiment import _aligned_cache


# Audited input snapshot before the September 16 search; paths are relative to artifacts.
PINNED_INPUTS = {
    "experiments/patch_mixture/summary.json": "ece09f86739325f0f6ad438ca24d6072dd371ae0e6b0beae101df9346d70d49a",
    "experiments/physical_photo/summary.json": "3b054b9f5849b380a56408a8bb9c021402d0ba8d4c1e0a002381a32cef36474f",
    "experiments/physical_photo/oof_predictions.csv": "2e1f45153fc77241b9bb7f2050ca148f80178b0d94e7cc71d8637cd73144b6cc",
    "experiments/physical_photo/camera_predictions.csv": "a267cf0cd47f6096dcff845078bb713f669c5fe68654cff1919c3fa27c63cd0b",
    "experiments/dino_pca/summary.json": "288bc83a1c3282e2c0bfd4a575ccdd7ccf33db2533dfac561ed0b6daaa4e7b05",
    "experiments/dino_pca/oof_predictions.csv": "36e4c13a8cc728ebc2669320d68cb6a1d647b44f307f2cafada4faeba9811ac4",
    "experiments/dino_pca/camera_predictions.csv": "4398d56f4a9cb47fce88655cb7fbdefd62aeadcd85b19944d35b4e2f23ac2298",
    "experiments/mobilenet_pca/summary.json": "15d1e654c453de16aab1b201ce41b29ff5ac1c48f2309759907188efb4e77e63",
    "experiments/mobilenet_pca/oof_predictions.csv": "bac3a2804a4fdfea11a41a5b0a9aa665289d7765a0bd2c2ae33a3d79fa2e5127",
    "experiments/mobilenet_pca/camera_predictions.csv": "90b11812bd2682541dfe99005c2d56d5e54be7ae30a591c98056a2870c239a70",
}


def _verified_sources(records: list[dict]) -> list[dict]:
    verified = {}

    def inspect(node):
        if isinstance(node, list):
            for value in node:
                inspect(value)
        elif isinstance(node, dict):
            if "path" in node and "sha256" in node:
                path = Path(node["path"])
                key = path.resolve()
                actual = (verified[key]["sha256"] if key in verified
                          else sha256(path.read_bytes()).hexdigest())
                if actual != node["sha256"]:
                    raise ValueError(f"Recorded source changed: {path}")
                if key not in verified:
                    verified[key] = {"path": str(path), "sha256": actual}
                    if path.suffix == ".json":
                        inspect(json.loads(path.read_text()))
            if "checkpoint_path" in node and "checkpoint_sha256" in node:
                inspect({"path": node["checkpoint_path"], "sha256": node["checkpoint_sha256"]})
            for value in node.values():
                inspect(value)

    inspect(records)
    return list(verified.values())


def load_search_inputs(config_path: str | Path = "configs/data.yaml"):
    """Load the declared batch's verified, aligned inputs without fitting or scoring."""
    cfg = load_config(config_path)
    truth, sample, ppm = [pd.read_csv(cfg.curated_file(key)) for key in ("train", "sample_submission", "ppm")]
    validate_cumulative_curves(truth)
    for frame in (truth, sample):
        if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError("Labels/template require nonempty, unique, complete soil IDs")
    if len(truth) < 3 or set(truth["sample_id"]) & set(sample["sample_id"]):
        raise ValueError("Search requires at least three training soils and disjoint test soils")
    keys = ["split", "sample_id", "camera", "path"]
    index_path = cfg.reports_dir / "photo_index.csv"
    index = pd.read_csv(index_path)[keys]
    if index.isna().any().any() or index["path"].duplicated().any() or set(index["split"]) != {"train", "test"}:
        raise ValueError("Photo index requires complete keys, unique paths, and train/test splits")
    for split, frame in (("train", truth), ("test", sample)):
        if set(index.loc[index["split"] == split, "sample_id"]) != set(frame["sample_id"]):
            raise ValueError(f"Photo index {split} coverage differs from labels/template")
    if (ppm["phone"].isna().any() or ppm["phone"].duplicated().any()
            or not set(index["camera"]).issubset(set(ppm["phone"]))):
        raise ValueError("Camera calibration requires unique complete phones covering every photo")
    sources = _verified_sources([
        {"path": str(cfg.artifacts_dir / path), "sha256": digest} for path, digest in PINNED_INPUTS.items()
    ])
    verified_paths = {Path(item["path"]).resolve() for item in sources}

    def require_verified(path):
        if Path(path).resolve() not in verified_paths:
            raise ValueError(f"Input missing from recorded sources: {path}")

    for path in [config_path, *(cfg.curated_file(key) for key in ("train", "sample_submission", "ppm")),
                 index_path, *index["path"]]:
        require_verified(path)

    photos_by_name = {}
    for name, directory, count, retained, dtype in (
        ("spectral", "patch_mixture", 29, 23, np.float64), ("dino", "dino_pca", 384, 384, np.float32),
    ):
        path = cfg.artifacts_dir / "experiments" / directory / "photo_features.csv"
        require_verified(path)
        columns = [f"feature_{i}" for i in range(retained)]
        photos_by_name[name] = _aligned_cache(path, index, count)[keys + columns].astype({c: dtype for c in columns})

    components = {}
    expected_views = index.loc[index["split"] == "train", ["sample_id", "camera"]].drop_duplicates()
    for name, directory, candidate in (
        ("spectral", "physical_photo", "rgb_texture_spectral_ridge.csv"),
        ("dino", "dino_pca", "dinov2_vits14_patch_pca8_nested_ridge.csv"),
        ("mobilenet", "mobilenet_pca", "mobilenet_v3_large_pca8_nested_ridge.csv"),
    ):
        component = {}
        for kind, expected, prediction_keys in (
            ("oof", truth, ["sample_id"]), ("camera", expected_views, ["sample_id", "camera"]),
        ):
            path = cfg.artifacts_dir / "experiments" / directory / f"{kind}_predictions.csv"
            require_verified(path)
            frame = pd.read_csv(path)
            if name == "spectral":
                frame = frame.loc[frame["experiment"] == "spectral_ridge"]
            component[kind] = _aligned_predictions(frame, expected, prediction_keys)
        path = cfg.submissions_dir / candidate
        require_verified(path)
        frame = pd.read_csv(path)
        validate_submission(frame, sample)
        component["submission"] = _aligned_predictions(frame, sample, ["sample_id"])
        components[name] = component
    return truth, sample, photos_by_name, components, sources
