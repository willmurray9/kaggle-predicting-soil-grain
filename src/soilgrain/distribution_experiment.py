"""Four fixed September 18 target/texture comparisons; no automatic uploads."""
from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.camera_transfer import evaluate_camera_transfer, summarize_camera_transfer
from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.experiments import evaluate_samples
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.pca_experiment import _camera_disagreement
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves


def record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def native_feature_cache(index: pd.DataFrame, cameras: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    from soilgrain import native_texture

    keys = ["split", "sample_id", "camera", "path"]
    index = index[keys].reset_index(drop=True)
    if index.isna().any().any() or index.path.duplicated().any():
        raise ValueError("Require unique complete photo keys")
    identity = {"photo_keys": index.to_dict("records"), "cameras": cameras.to_json(),
                "spectral_bands_mm": [list(band) for band in native_texture.SPECTRAL_BANDS_MM],
                "recipe_sha256": sha256(Path(native_texture.__file__).read_bytes()).hexdigest()}
    manifest_path = output / "native_cache_manifest.json"
    cached = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if cached and cached["identity"] != identity:
        raise ValueError("Native cache inputs or recipe changed")
    frames, records = {}, {}
    for name, count, extractor in (
        ("native_spectral", 23, native_texture.native_spectral_features),
        ("lbp", 40, native_texture.physical_lbp_features),
    ):
        path = output / f"{name}_photo_features.csv"
        if cached:
            if record(path) != cached["outputs"][name]:
                raise ValueError("Native feature cache changed")
            frame = pd.read_csv(path)
        else:
            columns = [f"feature_{i}" for i in range(count)]
            values = np.vstack([extractor(Path(row.path), cameras.loc[row.camera])
                                for row in index.itertuples(index=False)])
            frame = pd.concat([index, pd.DataFrame(values, columns=columns)], axis=1)
            write_csv(frame, path)
        if (not frame[keys].equals(index) or frame.shape != (len(index), count + 4)
                or not np.isfinite(frame.iloc[:, 4:].to_numpy(dtype=float)).all()):
            raise ValueError("Invalid native feature cache alignment or values")
        frames[name], records[name] = frame, record(path)
    if not cached:
        write_json({"identity": identity, "outputs": records}, manifest_path)
    return frames


def save_candidate(name, photos, predictor, truth, sample, reference, output: Path) -> dict:
    ids, curves = truth.sample_id.tolist(), curve_array(truth)
    train = photos.loc[photos.split == "train"]
    columns = [c for c in photos if c.startswith("feature_")]
    oof, camera = evaluate_samples(train, ids, curves, predictor)
    validate_cumulative_curves(pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS))
    errors = np.array([emd_score(y, p) for y, p in zip(curves, oof, strict=True)])
    baseline = curve_array(reference.set_index("sample_id").loc[ids])
    ref_errors = np.array([emd_score(y, p) for y, p in zip(curves, baseline, strict=True)])
    improvement = ref_errors - errors
    oof_frame = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    oof_frame.insert(0, "sample_id", ids)
    oof_frame["emd"] = errors
    oof_frame["reference_emd"] = ref_errors
    contributions = np.abs(curves[:, :10] - oof[:, :10]) * np.diff(np.log10(SUPPORT_DIAMETERS))
    oof_frame["fine_emd_at_or_below_0_2mm"] = contributions[:, :5].sum(axis=1)
    oof_frame["improvement_emd"] = improvement
    means = photos.groupby(["split", "sample_id"])[columns].mean()
    candidate = sample.copy()
    candidate[ordered_grain_columns(sample)] = predictor(
        means.loc[("train", ids), :].to_numpy(), curves,
        means.loc[("test", sample.sample_id.tolist()), :].to_numpy(),
    )
    validate_submission(candidate, sample)
    transfer = evaluate_camera_transfer(photos, ids, curves, predictor)
    directions, pairs = summarize_camera_transfer(transfer)
    path = output / name
    frames = {"oof": oof_frame, "camera": camera, "camera_transfer": transfer,
              "directions": directions, "pairs": pairs, "submission": candidate}
    files = {kind: record(write_csv(frame, path / f"{kind}.csv")) for kind, frame in frames.items()}
    disagreement, camera_pairs = _camera_disagreement(camera)
    result = {
        "experiment": name, "feature_count": len(columns), "loo_emd": float(errors.mean()),
        "paired_camera_disagreement_emd": disagreement, "camera_pairs": camera_pairs,
        "soils_improved": int((improvement > 0).sum()), "mean_improvement_emd": float(improvement.mean()),
        "median_improvement_emd": float(np.median(improvement)),
        "largest_beneficiary": ids[int(np.argmax(improvement))],
        "mean_improvement_excluding_largest": float(np.delete(improvement, np.argmax(improvement)).mean()),
        "fine_emd_at_or_below_0_2mm": float(contributions[:, :5].sum(axis=1).mean()),
        "transfer_directions": directions.to_dict("records"), **files,
    }
    print(f"{name}: {result['loo_emd']:.5f} EMD; camera {disagreement:.5f}", flush=True)
    return result


def run_distribution_search(config_path: str = "configs/data.yaml") -> dict:
    from soilgrain.config import load_config
    from soilgrain.distribution_models import sqrt_mass_ridge, emd_median_regression
    from soilgrain.search_inputs import load_search_inputs

    cfg = load_config(config_path)
    truth, sample, photos, components, sources = load_search_inputs(config_path)
    output = cfg.artifacts_dir / "experiments" / "distribution_search"
    cameras = pd.read_csv(cfg.curated_file("ppm")).set_index("phone")
    native = native_feature_cache(photos["spectral"], cameras, output)
    candidates = [save_candidate(name, frame, predictor, truth, sample,
                                 components["spectral"]["oof"], output)
                  for name, frame, predictor in (
        ("sqrt_mass_ridge", photos["spectral"], sqrt_mass_ridge),
        ("emd_median_regression", photos["spectral"], emd_median_regression),
        ("native_spectral_ridge", native["native_spectral"], ridge_curves),
        ("lbp_ridge", native["lbp"], ridge_curves),
    )]
    for name in ("distribution_models.py", "distribution_experiment.py", "native_texture.py",
                 "camera_transfer.py", "ridge.py", "metrics.py", "spectral_features.py"):
        sources.append(record(Path(__file__).with_name(name)))
    sources.append(record(output / "native_cache_manifest.json"))
    for name in native:
        sources.append(record(output / f"{name}_photo_features.csv"))
    summary = {"producing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "declaration": record(Path("docs/september18-plan.md")), "sources": sources,
               "candidates": candidates, "train_soils": len(truth), "test_soils": len(sample),
               "selection": "Four fixed recipes; no parameter search or automatic upload"}
    write_json(summary, output / "summary.json")
    return summary
