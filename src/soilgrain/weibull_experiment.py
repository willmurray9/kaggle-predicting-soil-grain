from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.experiments import evaluate_samples
from soilgrain.io import write_csv, write_json
from soilgrain.metrics import emd_score
from soilgrain.ridge import ridge_curves
from soilgrain.search_inputs import PINNED_INPUTS, _verified_sources, load_search_inputs
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.weibull import FIT_RECIPE, decode_weibull, fit_weibull, log_parameter_ridge


CAPACITY_GATE = {"mean_emd_at_most": 5., "max_emd_at_most": 15.}


def _scored_frame(sample_ids, truth, predictions):
    frame = pd.DataFrame(predictions, columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, "sample_id", sample_ids)
    contributions = np.abs(truth[:, :10] - predictions[:, :10]) * np.diff(np.log10(SUPPORT_DIAMETERS))
    frame["emd"] = [emd_score(y, p) for y, p in zip(truth, predictions, strict=True)]
    frame["fine_emd"] = contributions[:, :4].sum(axis=1)
    for i, label in enumerate(CANONICAL_GRAIN_LABELS[:10]):
        frame[f"emd_contribution_{label}"] = contributions[:, i]
    validate_cumulative_curves(frame)
    return frame


def _records(paths):
    return [{"path": str(path), "sha256": sha256(Path(path).read_bytes()).hexdigest()} for path in paths]


def run_weibull(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    """Test representation capacity first; regress only after the frozen gate passes."""
    cfg = load_config(config_path)
    manifest_key = "experiments/physical_photo/summary.json"
    label_manifest = {"path": str(cfg.artifacts_dir / manifest_key), "sha256": PINNED_INPUTS[manifest_key]}
    verified_labels = _verified_sources([label_manifest])
    verified_paths = {Path(record["path"]).resolve() for record in verified_labels}
    for path in (Path(config_path), cfg.curated_file("train")):
        if path.resolve() not in verified_paths:
            raise ValueError(f"Label input missing from recorded sources: {path}")
    producing_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    truth = pd.read_csv(cfg.curated_file("train"))
    validate_cumulative_curves(truth)
    if (len(truth) < 3 or truth.sample_id.isna().any() or truth.sample_id.duplicated().any()):
        raise ValueError("Require at least three unique complete training soil IDs")
    ids, curves = truth.sample_id.tolist(), curve_array(truth)
    parameters = np.vstack([fit_weibull(curve) for curve in curves])
    reconstructed = _scored_frame(ids, curves, decode_weibull(parameters))
    fitted = pd.DataFrame({"sample_id": ids, "log_scale": parameters[:, 0], "log_shape": parameters[:, 1],
                           "scale_mm": np.exp(parameters[:, 0]), "shape": np.exp(parameters[:, 1])})
    mean_error, max_error = float(reconstructed.emd.mean()), float(reconstructed.emd.max())
    gate_passed = mean_error <= CAPACITY_GATE["mean_emd_at_most"] and max_error <= CAPACITY_GATE["max_emd_at_most"]
    output = cfg.artifacts_dir / "experiments" / "weibull"
    paths = {"parameters": write_csv(fitted, output / "label_parameters.csv"),
             "reconstruction": write_csv(reconstructed, output / "label_reconstruction.csv")}
    sources = _records([config_path, cfg.curated_file("train"), Path(__file__),
                        Path(__file__).with_name("weibull.py")]) + [label_manifest]
    summary = {
        "experiment": "spectral23_weibull_ridge", "fit_recipe": FIT_RECIPE, "producing_commit": producing_commit,
        "capacity_gate": CAPACITY_GATE, "capacity_gate_passed": bool(gate_passed),
        "reconstruction_mean_emd": mean_error, "reconstruction_max_emd": max_error,
        "soils_above_maximum": reconstructed.loc[reconstructed.emd > 15., "sample_id"].tolist(),
        "train_samples": len(ids), "fine_supports_mm": list(SUPPORT_DIAMETERS[:4]),
        "limitation": "Reconstruction is approximation error, not predictive validation or a certified global optimum. The known-label capacity gate is exploratory model selection.",
        "sources": sources,
    }
    print(f"Weibull reconstruction: mean {mean_error:.4f}, max {max_error:.4f} EMD; gate passed: {gate_passed}", flush=True)
    if not gate_passed:
        summary["status"] = "stopped_at_capacity_gate"
        summary["outputs"] = _records(paths.values())
        paths["summary"] = write_json(summary, output / "summary.json")
        return paths

    # Each label fit is independent; the fold adapter selects only supplied training labels.
    verified_truth, sample, photos_by_name, components, verified_sources = load_search_inputs(config_path)
    if verified_truth.sample_id.tolist() != ids or not np.array_equal(curve_array(verified_truth), curves):
        raise ValueError("Labels changed between capacity fitting and verified feature loading")
    lookup = {tuple(curve): parameter for curve, parameter in zip(curves, parameters, strict=True)}

    def predictor(train_features, train_curves, query_features):
        training_parameters = np.vstack([lookup[tuple(curve)] for curve in train_curves])
        return log_parameter_ridge(train_features, training_parameters, query_features)

    photos = photos_by_name["spectral"]
    train_photos = photos.loc[photos.split == "train"]
    columns = [f"feature_{i}" for i in range(23)]
    oof, camera = evaluate_samples(train_photos, ids, curves, predictor)
    predicted = _scored_frame(ids, curves, oof)
    reference = components["spectral"]["oof"].set_index("sample_id").loc[ids].reset_index()
    reference = _scored_frame(ids, curves, curve_array(reference))
    improvement = reference.emd.to_numpy() - predicted.emd.to_numpy()
    comparison = pd.DataFrame({"sample_id": ids, "reference_emd": reference.emd, "weibull_emd": predicted.emd,
                               "improvement_emd": improvement, "reference_fine_emd": reference.fine_emd,
                               "weibull_fine_emd": predicted.fine_emd})
    means = photos.groupby(["split", "sample_id"])[columns].mean()
    submission = sample.copy()
    submission[ordered_grain_columns(sample)] = log_parameter_ridge(
        means.loc[("train", ids), :].to_numpy(), parameters,
        means.loc[("test", sample.sample_id.tolist()), :].to_numpy(),
    )
    validate_submission(submission, sample)

    from soilgrain.camera_transfer import evaluate_camera_transfer, summarize_camera_transfer

    transfer = evaluate_camera_transfer(photos, ids, curves, predictor)
    reference_transfer = evaluate_camera_transfer(photos, ids, curves, ridge_curves)
    directions, paired = summarize_camera_transfer(transfer)
    keys = ["sample_id", "train_camera", "query_camera", "direction"]
    transfer_comparison = transfer[keys + ["emd", "fine_emd"]].merge(
        reference_transfer[keys + ["emd", "fine_emd"]], on=keys, validate="one_to_one", suffixes=("_weibull", "_reference"),
    )
    transfer_comparison["improvement_emd"] = transfer_comparison.emd_reference - transfer_comparison.emd_weibull
    paths.update({
        "oof": write_csv(predicted, output / "oof_predictions.csv"),
        "camera_predictions": write_csv(camera, output / "camera_predictions.csv"),
        "comparison": write_csv(comparison, output / "per_soil_comparison.csv"),
        "camera_transfer": write_csv(transfer, output / "camera_transfer_predictions.csv"),
        "camera_transfer_directions": write_csv(directions, output / "camera_transfer_directions.csv"),
        "camera_transfer_pairs": write_csv(paired, output / "camera_transfer_pairs.csv"),
        "camera_transfer_reference": write_csv(reference_transfer, output / "camera_transfer_reference.csv"),
        "camera_transfer_comparison": write_csv(transfer_comparison, output / "camera_transfer_comparison.csv"),
        "submission": write_csv(submission, output / "submission.csv"),
    })
    summary.update({
        "status": "evaluated", "test_samples": len(sample), "feature_count": 23, "ridge_alpha": 10.,
        "validation": "Whole-soil LOO; label fitting is independent per soil and regression/scaling consume only fold training soils. Paired camera folds exclude both held-out views.",
        "loo_emd": float(predicted.emd.mean()), "reference_loo_emd": float(reference.emd.mean()),
        "loo_fine_emd": float(predicted.fine_emd.mean()), "reference_loo_fine_emd": float(reference.fine_emd.mean()),
        "mean_improvement_emd": float(improvement.mean()), "median_improvement_emd": float(np.median(improvement)),
        "largest_beneficiary": ids[int(np.argmax(improvement))],
        "mean_improvement_excluding_largest_beneficiary": float(np.delete(improvement, np.argmax(improvement)).mean()),
        "camera_transfer_summary": transfer_comparison.groupby("direction")[[
            "emd_weibull", "emd_reference", "fine_emd_weibull", "fine_emd_reference", "improvement_emd",
        ]].mean().reset_index().to_dict("records"),
        "sources": sources + verified_sources + _records([Path(__file__).with_name("camera_transfer.py")]),
        "outputs": _records(paths.values()),
    })
    paths["summary"] = write_json(summary, output / "summary.json")
    print(f"Weibull spectral ridge: {summary['loo_emd']:.4f} EMD; reference {summary['reference_loo_emd']:.4f}", flush=True)
    return paths
