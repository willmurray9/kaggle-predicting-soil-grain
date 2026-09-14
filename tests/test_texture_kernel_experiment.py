import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.ridge import ridge_curves
from soilgrain.texture_kernel_experiment import write_texture_kernel_experiment


@pytest.fixture
def kernel_run(tmp_path):
    curated, artifacts = tmp_path / "curated", tmp_path / "artifacts"
    first, texture = [artifacts / "experiments" / name for name in ("first_batch", "physical_texture")]
    reports, submissions = artifacts / "reports", artifacts / "submissions"
    for directory in (curated, first, texture, reports, submissions):
        directory.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n  reports_dir: {reports}\n  submissions_dir: {submissions}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n"
    )
    ids = ["D", "B", "E", "A", "C"]
    curves = np.array([list(np.linspace(v, 90., 10)) + [100.] for v in (2., 30., 12., 55., 65.)])
    truth = pd.DataFrame(curves, columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0, "sample_id", ids)
    truth.to_csv(curated / "train.csv", index=False)
    sample = truth.iloc[:2].copy()
    sample["sample_id"] = ["T2", "T1"]
    sample.to_csv(curated / "sample.csv", index=False)
    index = pd.DataFrame([
        {"split": "test" if soil.startswith("T") else "train", "sample_id": soil,
         "camera": camera, "path": f"{soil}_{camera}_{i}.jpg"}
        for soil in [*ids, "T1", "T2"] for i, camera in enumerate(("one", "one", "two"))
    ])
    index.to_csv(reports / "photo_index.csv", index=False)
    features = np.random.default_rng(92).normal(size=(len(index), 17))
    columns = [f"feature_{i}" for i in range(17)]
    photos = pd.concat([index, pd.DataFrame(features, columns=columns)], axis=1)
    photos[[*index.columns, *columns[:13]]].to_csv(first / "reference_rgb_100_photo_features.csv", index=False)
    extras = pd.concat([index, pd.DataFrame(features[:, 13:], columns=[f"texture_{i}" for i in range(4)])], axis=1)
    extras.iloc[::-1].to_csv(texture / "photo_texture_features.csv", index=False)
    train_photos = photos[photos["split"] == "train"]
    oof, views = evaluate_samples(train_photos, ids, curves, ridge_curves)
    reference = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    reference.insert(0, "sample_id", ids)
    reference["emd"] = 999.
    for frame, name in ((reference, "oof"), (views, "camera")):
        frame.insert(0, "experiment", "ridge_rgb_texture_100")
        frame.iloc[::-1].to_csv(texture / f"{name}_predictions.csv", index=False)
    train = train_photos.groupby("sample_id")[columns].mean().loc[ids].to_numpy()
    test = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[["T2", "T1"]].to_numpy()
    sample[list(CANONICAL_GRAIN_LABELS)] = ridge_curves(train, curves, test)
    sample.iloc[::-1].to_csv(submissions / "ridge_rgb_texture_100.csv", index=False)
    (texture / "summary.json").write_text(json.dumps({
        "sources": [{"path": str(curated / "train.csv"), "sha256": sha256((curated / "train.csv").read_bytes()).hexdigest()}],
    }))
    return config, artifacts, photos


def test_kernel_experiment_aligns_features_reconstructs_incumbent_and_keeps_inputs(kernel_run):
    config, artifacts, expected_photos = kernel_run
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    paths = write_texture_kernel_experiment(config)
    summary = json.loads(paths["summary"].read_text())
    assert max(summary["reference_max_abs_differences"].values()) < 1e-10
    pd.testing.assert_frame_equal(pd.read_csv(paths["features"]), expected_photos, rtol=1e-12, atol=1e-12)
    assert summary["feature_count"] == 17
    assert len(pd.read_csv(paths["outer_selection"])) == 30
    assert len(pd.read_csv(paths["final_selection"])) == 6
    assert set(pd.read_csv(paths["oof"])["experiment"]) == {"fixed", "nested"}
    assert pd.read_csv(paths["submission"])["sample_id"].tolist() == ["T2", "T1"]
    assert all(p.read_bytes() == value for p, value in before.items())
    expected = str(paths["submission"]) if summary["submission_evidence"]["eligible"] else None
    assert summary["selected_submission"] == expected
    for item in [*summary["sources"], summary["submission"], summary["feature_cache"]]:
        assert sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.parametrize("corruption", ["texture_key", "texture_value", "reference", "label_source"])
def test_kernel_experiment_rejects_changed_features_or_references_before_candidates(kernel_run, corruption):
    config, artifacts, _photos = kernel_run
    texture = artifacts / "experiments" / "physical_texture"
    if corruption == "label_source":
        path = config.parent / "curated" / "train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "0.002"] += 0.01
    elif corruption == "reference":
        path = texture / "oof_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "0.002"] -= 0.01
    else:
        path = texture / "photo_texture_features.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "camera" if corruption == "texture_key" else "texture_0"] = "wrong" if corruption == "texture_key" else 1e5
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="coverage|reference|source changed"):
        write_texture_kernel_experiment(config)
    assert not (artifacts / "experiments" / "nested_texture_kernel").exists()


def test_test_photos_cannot_change_validation_or_parameter_selection(kernel_run):
    config, artifacts, photos = kernel_run
    paths = write_texture_kernel_experiment(config)
    retained = {key: paths[key].read_bytes() for key in ("oof", "camera_predictions", "outer_selection", "final_selection", "comparison")}
    columns = [f"feature_{i}" for i in range(17)]
    keys = ["split", "sample_id", "camera", "path"]
    photos.loc[photos["split"] == "test", columns] *= 1e6
    photos[[*keys, *columns[:13]]].to_csv(artifacts / "experiments" / "first_batch" / "reference_rgb_100_photo_features.csv", index=False)
    photos[[*keys, *columns[13:]]].rename(columns={f"feature_{i + 13}": f"texture_{i}" for i in range(4)}).to_csv(
        artifacts / "experiments" / "physical_texture" / "photo_texture_features.csv", index=False,
    )
    truth = pd.read_csv(config.parent / "curated" / "train.csv")
    sample = pd.read_csv(config.parent / "curated" / "sample.csv")
    train = photos[photos["split"] == "train"].groupby("sample_id")[columns].mean().loc[truth["sample_id"]].to_numpy()
    test = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[sample["sample_id"]].to_numpy()
    sample[list(CANONICAL_GRAIN_LABELS)] = ridge_curves(train, truth[list(CANONICAL_GRAIN_LABELS)].to_numpy(), test)
    sample.to_csv(artifacts / "submissions" / "ridge_rgb_texture_100.csv", index=False)
    write_texture_kernel_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())


def test_perfect_fixed_diagnostic_cannot_authorize_failing_nested_candidate(kernel_run, monkeypatch):
    import soilgrain.texture_kernel_experiment as experiment

    config, _artifacts, _photos = kernel_run
    original_fixed, original_nested = experiment.evaluate_samples, experiment.evaluate_nested_kernel

    def fixed(photos, ids, curves, predictor):
        oof, views = original_fixed(photos, ids, curves, predictor)
        if predictor is experiment.kernel_ridge_curves:
            oof = curves.copy()
            truth = pd.DataFrame(curves, index=ids, columns=CANONICAL_GRAIN_LABELS)
            views[list(CANONICAL_GRAIN_LABELS)] = truth.loc[views["sample_id"]].to_numpy()
        return oof, views

    def nested(photos, ids, curves):
        oof, views, selections = original_nested(photos, ids, curves)
        oof[:, :10] = 0.
        oof[:, 10] = 100.
        views[list(CANONICAL_GRAIN_LABELS)] = [0.] * 10 + [100.]
        return oof, views, selections

    monkeypatch.setattr(experiment, "evaluate_samples", fixed)
    monkeypatch.setattr(experiment, "evaluate_nested_kernel", nested)
    paths = write_texture_kernel_experiment(config)
    summary = json.loads(paths["summary"].read_text())
    fixed_result, nested_result = summary["experiments"]
    assert fixed_result["outer_loo_emd"] == 0.
    assert nested_result["outer_loo_emd"] > summary["incumbent_emd"]
    assert summary["submission_evidence"]["mean_improvement_emd"] < 0.
    assert summary["selected_submission"] is None
