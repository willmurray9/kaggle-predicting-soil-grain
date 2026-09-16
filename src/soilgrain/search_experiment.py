from __future__ import annotations

import subprocess
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.io import write_csv, write_json
from soilgrain.model_blend import _blend_predictions
from soilgrain.multicrop import _score_rows
from soilgrain.nested_ridge import evaluate_nested_ridge, select_ridge_alpha
from soilgrain.pca_experiment import _camera_disagreement, submission_evidence
from soilgrain.ridge import ridge_curves
from soilgrain.submission import validate_submission
from soilgrain.targets import curve_array, ordered_grain_columns, validate_cumulative_curves
from soilgrain.texture_experiment import _reference_difference

PRIOR_SUBMISSIONS = (
    "multicrop_baseline.csv", "experiments/ridge_rgb_100.csv", "frozen_resnet18_nested_ridge.csv",
    "spatial_multicrop.csv", "ridge_rgb_texture_100.csv", "ridge_gray_100.csv",
    "ridge_gray_texture_100.csv", "rgb_texture_pca8_blend.csv", "rgb_texture_official_pca8_blend.csv",
    "rgb_texture_spectral_ridge.csv", "rgb_texture_spectral_photo_ridge.csv",
)


def _submission_queue(rows, prior_paths):
    """Rank declared candidates and skip previously tried or duplicate CDFs."""
    seen, queue, skipped = [], [], []
    template = None
    for path in prior_paths:
        frame = pd.read_csv(path)
        template = frame if template is None else template
        validate_submission(frame, template)
        seen.append((str(path), curve_array(frame.set_index("sample_id").sort_index())))
    for row in sorted(rows, key=lambda r: (r["loo_emd"], r["paired_camera_disagreement_emd"], r["experiment"])):
        frame = pd.read_csv(row["submission"]["path"])
        template = frame if template is None else template
        validate_submission(frame, template)
        values = curve_array(frame.set_index("sample_id").sort_index())
        duplicate = next((name for name, old in seen if np.allclose(values, old, rtol=0, atol=1e-9)), None)
        if duplicate is not None:
            skipped.append({"experiment": row["experiment"], "duplicate_of": duplicate})
        else:
            queue.append(row["experiment"])
            seen.append((row["experiment"], values))
    return queue, skipped


def write_search_experiment(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    from soilgrain.boost_model import boost_curves
    from soilgrain.pls import evaluate_nested_pls, pls_curves, select_pls_components
    from soilgrain.search_inputs import load_search_inputs

    cfg = load_config(config_path)
    truth, sample, photos, components, sources = load_search_inputs(config_path)
    train_ids, test_ids = truth["sample_id"].tolist(), sample["sample_id"].tolist()
    curves = curve_array(truth)
    output = cfg.artifacts_dir / "experiments" / "autonomous_search"
    frames, matrices = {}, {}
    for name, frame in photos.items():
        columns = [c for c in frame if c.startswith("feature_")]
        frames[name] = frame.loc[frame["split"] == "train"]
        matrices[name] = (
            frames[name].groupby("sample_id")[columns].mean().loc[train_ids].to_numpy(),
            frame.loc[frame["split"] == "test"].groupby("sample_id")[columns].mean().loc[test_ids].to_numpy(),
        )

    def predictions(values):
        result = pd.DataFrame(values, columns=CANONICAL_GRAIN_LABELS)
        result.insert(0, "sample_id", train_ids)
        return result

    def submission(values):
        result = sample.copy()
        result[ordered_grain_columns(sample)] = values
        validate_submission(result, sample)
        return result

    reference = components["spectral"]
    oof, views = evaluate_samples(frames["spectral"], train_ids, curves, ridge_curves)
    reconstructed = {"oof": predictions(oof), "camera": views,
                     "submission": submission(ridge_curves(matrices["spectral"][0], curves, matrices["spectral"][1]))}
    differences = {kind: _reference_difference(frame, reference[kind],
                   ["sample_id", "camera"] if kind == "camera" else ["sample_id"])
                   for kind, frame in reconstructed.items()}
    reference_errors = _score_rows(truth, reference["oof"])
    reference_camera, _ = _camera_disagreement(reference["camera"])
    print(f"Verified spectral reference: {reference_errors.mean():.5f} EMD.", flush=True)
    rows, failures = [], []

    def save(name, result):
        for kind in ("oof", "camera"):
            validate_cumulative_curves(result[kind])
            result[kind] = result[kind].copy()
            result[kind]["emd"] = _score_rows(truth, result[kind])
        validate_submission(result["submission"], sample)
        errors = result["oof"]["emd"].to_numpy()
        camera, pairs = _camera_disagreement(result["camera"])
        paths = {kind: write_csv(frame, output / name / f"{name if kind == 'submission' else kind}.csv")
                 for kind, frame in result.items()}
        row = {"experiment": name, "loo_emd": float(errors.mean()),
               "paired_camera_disagreement_emd": camera, "camera_pairs": pairs,
               "incumbent_screen_diagnostic": submission_evidence(errors, reference_errors, camera, reference_camera),
               **{kind: {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()} for kind, path in paths.items()}}
        rows.append(row)
        print(f"{name}: {errors.mean():.5f} EMD; camera {camera:.5f}.", flush=True)

    for name in ("dino", "mobilenet"):
        saved = components[name]
        save(f"{name}_pca", saved.copy())
        save(f"spectral_{name}_blend", {
            kind: _blend_predictions(reference[kind], saved[kind],
                                     ["sample_id", "camera"] if kind == "camera" else ["sample_id"])
            for kind in ("oof", "camera", "submission")
        })

    for name, feature_name, evaluator, selector, predictor, parameter in (
        ("dino_pls", "dino", evaluate_nested_pls, select_pls_components, pls_curves, "n_components"),
        ("spectral_nested_ridge", "spectral", evaluate_nested_ridge, select_ridge_alpha, ridge_curves, "alpha"),
        ("spectral_boost", "spectral", None, None, boost_curves, None),
    ):
        try:
            train, test = matrices[feature_name]
            selection_paths = {}
            if evaluator is None:
                oof, views = evaluate_samples(frames[feature_name], train_ids, curves, predictor)
                options = {}
            else:
                oof, views, selection = evaluator(frames[feature_name], train_ids, curves)
                selected, scores = selector(train, curves)
                options = {parameter: selected}
                selection_paths["outer_selection"] = write_csv(selection, output / name / "outer_selection.csv")
                selection_paths["final_selection"] = write_csv(
                    pd.DataFrame([{parameter: value, "loo_selection_emd": score, "selected": value == selected}
                                  for value, score in scores.items()]), output / name / "final_selection.csv")
            save(name, {"oof": predictions(oof), "camera": views,
                        "submission": submission(predictor(train, curves, test, **options))})
            rows[-1].update({kind: {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
                             for kind, path in selection_paths.items()})
        except (ConvergenceWarning, FloatingPointError, np.linalg.LinAlgError) as error:
            failures.append({"experiment": name, "error": str(error)})
            print(f"Excluded {name}: {error}", flush=True)

    prior_paths = [cfg.submissions_dir / name for name in PRIOR_SUBMISSIONS]
    queue, skipped = _submission_queue(rows, prior_paths)
    producing_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    extra_sources = [*prior_paths, Path("docs/autonomous-search-plan.md"), Path(__file__),
                     Path("src/soilgrain/pls.py"), Path("src/soilgrain/boost_model.py"), Path("src/soilgrain/search_inputs.py")]
    sources += [{"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()} for path in extra_sources]
    paths = {"ranking": write_csv(pd.DataFrame([{k: row[k] for k in ("experiment", "loo_emd", "paired_camera_disagreement_emd")}
                                               for row in rows]).sort_values(["loo_emd", "paired_camera_disagreement_emd", "experiment"]),
                                   output / "ranking.csv")}
    paths["summary"] = write_json({
        "producing_commit": producing_commit, "plan": "docs/autonomous-search-plan.md", "target_public_emd": 55.78511,
        "validation": "Whole-soil LOO; nested training-only selection for PLS/ridge; fixed 50/50 aligned blends",
        "selection_note": "Exploratory batch ranking; final-selection scores are not validation estimates",
        "train_samples": len(truth), "test_samples": len(sample), "reference_max_abs_differences": differences,
        "incumbent_emd": float(reference_errors.mean()), "incumbent_camera_disagreement_emd": reference_camera,
        "candidates": rows, "submission_queue": queue, "duplicate_candidates": skipped, "failed_candidates": failures,
        "submission_rule": "Queue order fixed before upload; stop below55.78511 or when daily allowance exhausted",
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")},
        "sources": sources,
    }, output / "summary.json")
    return paths
