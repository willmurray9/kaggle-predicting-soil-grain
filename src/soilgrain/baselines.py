from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.data import load_working_tables
from soilgrain.io import ensure_dir, write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.submission import submission_from_curve, validate_submission
from soilgrain.targets import curve_array


def equal_bin_curve(n_bins: int = 11) -> np.ndarray:
    curve = np.cumsum(np.full(n_bins, 100.0 / n_bins))
    curve[-1] = 100.0
    return curve


def train_mean_curve(train: pd.DataFrame) -> np.ndarray:
    curve = curve_array(train).mean(axis=0)
    curve[-1] = 100.0
    return curve


def train_median_curve(train: pd.DataFrame) -> np.ndarray:
    curve = np.median(curve_array(train), axis=0)
    curve[-1] = 100.0
    return curve


def leave_one_out_score(train: pd.DataFrame, method: str) -> float | None:
    values = curve_array(train)
    if len(values) < 2:
        return None
    preds = []
    for i in range(len(values)):
        others = np.delete(values, i, axis=0)
        if method == "mean":
            pred = others.mean(axis=0)
        elif method == "median":
            pred = np.median(others, axis=0)
        else:
            raise ValueError(f"Unknown LOO baseline method: {method}")
        pred[-1] = 100.0
        preds.append(pred)
    return emd_score(values, np.vstack(preds))


def write_baseline_submissions(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train, _test, sample, _ppm = load_working_tables(cfg)
    ensure_dir(cfg.submissions_dir)

    curves = {
        "equal_bin": equal_bin_curve(),
        "mean": train_mean_curve(train),
        "median": train_median_curve(train),
    }
    paths: dict[str, Path] = {}
    for name, curve in curves.items():
        submission = submission_from_curve(sample, curve)
        validate_submission(submission, sample)
        paths[name] = write_csv(submission, cfg.submissions_dir / f"{name}_baseline.csv")

    train_values = curve_array(train)
    manifest = {
        "equal_bin_train_score": emd_score(train_values, np.tile(curves["equal_bin"], (len(train), 1))),
        "mean_leave_one_out_score": leave_one_out_score(train, "mean"),
        "median_leave_one_out_score": leave_one_out_score(train, "median"),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    write_json(manifest, cfg.reports_dir / "baseline_summary.json")
    return paths
