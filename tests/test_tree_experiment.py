import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.ridge import ridge_curves
from soilgrain.tree_experiment import write_tree_experiment


@pytest.fixture
def tree_run(tmp_path):
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
    ids = ["F", "B", "E", "A", "C", "D"]
    curves = np.array([[v] * 10 + [100.] for v in (10., 20., 30., 70., 80., 90.)])
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


def test_tree_experiment_reconstructs_reference_and_preserves_inputs(tree_run):
    config, artifacts, _photos = tree_run
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    paths = write_tree_experiment(config)
    summary = json.loads(paths["summary"].read_text())
    assert max(summary["reference_max_abs_differences"].values()) < 1e-10
    assert summary["feature_count"] == 17
    assert pd.read_csv(paths["submission"])["sample_id"].tolist() == ["T2", "T1"]
    assert summary["parameters"]["random_state"] == 42
    assert all(p.read_bytes() == value for p, value in before.items())
    # With only five training rows per outer fit, no split can leave three
    # soils on each side. The forest must predict the other soils' mean.
    predictions = pd.read_csv(paths["oof"])
    np.testing.assert_allclose(predictions["0.002"], [(300.-v)/5 for v in (10.,20.,30.,70.,80.,90.)])
    assert summary["selected_submission"] is None
    assert not summary["submission_evidence"]["eligible"]


@pytest.mark.parametrize("corruption", ["cache_hash", "photo_keys", "reference"])
def test_tree_experiment_rejects_changed_sources_or_predictions(tree_run, corruption):
    config, artifacts, _photos = tree_run
    cache_dir = artifacts / "experiments" / "nested_texture_kernel"
    if corruption == "reference":
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
        write_tree_experiment(config)
    assert not (artifacts / "experiments" / "shallow_trees").exists()
