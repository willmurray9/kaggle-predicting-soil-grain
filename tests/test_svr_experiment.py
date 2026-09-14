import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.ridge import ridge_curves
from soilgrain.svr_experiment import write_svr_experiment


@pytest.fixture
def svr_run(tmp_path):
    curated, artifacts = tmp_path / "curated", tmp_path / "artifacts"
    cache_dir, texture = [artifacts / "experiments" / name for name in ("nested_texture_kernel", "physical_texture")]
    reports, submissions = artifacts / "reports", artifacts / "submissions"
    for directory in (curated, cache_dir, texture, reports, submissions):
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
    columns = [f"feature_{i}" for i in range(17)]
    photos = pd.concat([index, pd.DataFrame(np.random.default_rng(92).normal(size=(len(index), 17)), columns=columns)], axis=1)
    cache = cache_dir / "photo_features.csv"
    photos.iloc[::-1].to_csv(cache, index=False)
    oof, views = evaluate_samples(photos[photos["split"] == "train"], ids, curves, ridge_curves)
    reference = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    reference.insert(0, "sample_id", ids)
    reference["emd"] = 999.
    for frame, name in ((reference, "oof"), (views, "camera")):
        frame.insert(0, "experiment", "ridge_rgb_texture_100")
        frame.iloc[::-1].to_csv(texture / f"{name}_predictions.csv", index=False)
    train = photos[photos["split"] == "train"].groupby("sample_id")[columns].mean().loc[ids].to_numpy()
    test = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[sample["sample_id"]].to_numpy()
    sample[list(CANONICAL_GRAIN_LABELS)] = ridge_curves(train, curves, test)
    sample.iloc[::-1].to_csv(submissions / "ridge_rgb_texture_100.csv", index=False)
    (cache_dir / "summary.json").write_text(json.dumps({
        "sources": [{"path": str(curated / "train.csv"), "sha256": sha256((curated / "train.csv").read_bytes()).hexdigest()}],
        "feature_cache": {"path": str(cache), "sha256": sha256(cache.read_bytes()).hexdigest()},
    }))
    return config, artifacts, photos


def test_svr_experiment_reconstructs_reference_and_preserves_inputs(svr_run):
    config, artifacts, _photos = svr_run
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    paths = write_svr_experiment(config)
    summary = json.loads(paths["summary"].read_text())
    assert max(summary["reference_max_abs_differences"].values()) < 1e-10
    assert summary["feature_count"] == 17
    assert len(pd.read_csv(paths["outer_selection"])) == 15
    assert len(pd.read_csv(paths["final_selection"])) == 3
    assert pd.read_csv(paths["submission"])["sample_id"].tolist() == ["T2", "T1"]
    assert summary["versions"]["scikit-learn"]
    assert all(p.read_bytes() == value for p, value in before.items())
    selected = str(paths["submission"]) if summary["submission_evidence"]["eligible"] else None
    assert summary["selected_submission"] == selected


@pytest.mark.parametrize("corruption", ["cache_hash", "photo_keys", "reference", "label_source"])
def test_svr_experiment_rejects_invalid_provenance_or_alignment(svr_run, corruption):
    config, artifacts, _photos = svr_run
    cache_dir = artifacts / "experiments" / "nested_texture_kernel"
    if corruption == "label_source":
        path = config.parent / "curated" / "train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "0.002"] += 0.01
    elif corruption == "reference":
        path = artifacts / "experiments" / "physical_texture" / "camera_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "0.002"] -= 0.01
    else:
        path = cache_dir / "photo_features.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "camera" if corruption == "photo_keys" else "feature_0"] = "wrong" if corruption == "photo_keys" else 1e5
    frame.to_csv(path, index=False)
    if corruption == "photo_keys":
        manifest_path = cache_dir / "summary.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["feature_cache"]["sha256"] = sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coverage|reference|source changed"):
        write_svr_experiment(config)
    assert not (artifacts / "experiments" / "linear_svr").exists()


def test_test_features_cannot_change_svr_validation_or_selection(svr_run):
    config, artifacts, photos = svr_run
    paths = write_svr_experiment(config)
    retained = {key: paths[key].read_bytes() for key in ("oof", "camera_predictions", "outer_selection", "final_selection", "comparison")}
    columns = [f"feature_{i}" for i in range(17)]
    photos.loc[photos["split"] == "test", columns] *= 1e6
    cache_dir = artifacts / "experiments" / "nested_texture_kernel"
    cache = cache_dir / "photo_features.csv"
    photos.to_csv(cache, index=False)
    manifest_path = cache_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_cache"]["sha256"] = sha256(cache.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    truth = pd.read_csv(config.parent / "curated" / "train.csv")
    sample = pd.read_csv(config.parent / "curated" / "sample.csv")
    train = photos[photos["split"] == "train"].groupby("sample_id")[columns].mean().loc[truth["sample_id"]].to_numpy()
    test = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[sample["sample_id"]].to_numpy()
    sample[list(CANONICAL_GRAIN_LABELS)] = ridge_curves(train, truth[list(CANONICAL_GRAIN_LABELS)].to_numpy(), test)
    sample.to_csv(artifacts / "submissions" / "ridge_rgb_texture_100.csv", index=False)
    write_svr_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())
