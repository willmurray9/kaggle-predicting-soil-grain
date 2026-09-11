import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.texture_experiment import write_texture_experiment


@pytest.fixture
def texture_run(tmp_path):
    curated, artifacts = tmp_path / "curated", tmp_path / "artifacts"
    first = artifacts / "experiments" / "first_batch"
    reports, submissions = artifacts / "reports", artifacts / "submissions"
    for directory in (curated, first, reports, submissions / "experiments"):
        directory.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n"
        f"  reports_dir: {reports}\n  submissions_dir: {submissions}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n  ppm: ppm.csv\n"
    )

    def curves(ids, values):
        result = pd.DataFrame({"sample_id": ids})
        result[list(CANONICAL_GRAIN_LABELS)] = [[v] * 10 + [100.] for v in values]
        return result

    curves(["B", "A", "D", "C"], [80, 20, 50, 50]).to_csv(curated / "train.csv", index=False)
    curves(["T2", "T1"], [0, 0]).to_csv(curated / "sample.csv", index=False)
    pd.DataFrame({"phone": ["one", "two"], "ppm": [2.56, 2.56], "width": [256, 256],
                  "height": [256, 256]}).to_csv(curated / "ppm.csv", index=False)
    checker = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(np.uint8)
    photos = []
    # A's three photos have descriptors [2,2,0,0], [0,0,0,0], [0,0,0,0].
    # Equal photo weighting gives 2/3, while equal camera weighting would give 1.
    for soil, camera, patterned in [("A", "one", True), ("A", "two", False), ("A", "two", False),
                                    ("B", "one", False), ("C", "one", True), ("D", "one", False),
                                    ("T1", "one", True), ("T2", "one", False)]:
        path = tmp_path / f"photo_{len(photos)}.png"
        Image.fromarray(checker if patterned else np.zeros_like(checker)).save(path)
        photos.append({"split": "test" if soil.startswith("T") else "train", "sample_id": soil,
                       "camera": camera, "path": str(path)})
    index = pd.DataFrame(photos)
    index.to_csv(reports / "photo_index.csv", index=False)
    for name, count in (("reference_rgb_100", 13), ("gray_100", 7)):
        cache = index.iloc[::-1].copy()
        cache[[f"feature_{i}" for i in range(count)]] = 0.
        cache.to_csv(first / f"{name}_photo_features.csv", index=False)
    reference = curves(["C", "D", "A", "B"], [50, 50, 60, 40])
    reference.insert(0, "experiment", "ridge_rgb_100")
    reference["emd"] = 999.
    reference.to_csv(first / "oof_predictions.csv", index=False)
    camera_reference = curves(["C", "D", "A", "A", "B"], [50, 50, 60, 60, 40])
    camera_reference.insert(0, "camera", ["one", "one", "two", "one", "one"])
    camera_reference.insert(0, "experiment", "ridge_rgb_100")
    camera_reference.to_csv(first / "camera_predictions.csv", index=False)
    curves(["T1", "T2"], [50, 50]).to_csv(submissions / "experiments" / "ridge_rgb_100.csv", index=False)
    return config, artifacts, first


def test_texture_run_aligns_caches_weights_photos_and_excludes_held_out_soils(texture_run, monkeypatch):
    from soilgrain.ridge import ridge_curves

    config, artifacts, _first = texture_run
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    calls = []

    def observe_ridge(features, targets, queries):
        calls.append((features.copy(), targets.copy(), queries.copy()))
        return ridge_curves(features, targets, queries)

    monkeypatch.setattr("soilgrain.texture_experiment.ridge_curves", observe_ridge)
    paths = write_texture_experiment(config)

    # Every outer fold for each configuration uses three soil rows, even when
    # the held-out soil has three photos and two cameras.
    for features, targets, queries in calls:
        if len(features) == 4:  # Final test fit.
            continue
        assert len(features) == 3
        if np.array_equal(targets[:, 0], [80, 50, 50]):  # A is held out.
            if features.shape[1] == 17:
                np.testing.assert_allclose(features[:, 13:], [[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0]])
                if len(queries) == 1:
                    np.testing.assert_allclose(queries[0, 13:], [2 / 3, 2 / 3, 0, 0])
                else:
                    np.testing.assert_allclose(queries[:, 13:], [[2, 2, 0, 0], [0, 0, 0, 0]])
    assert {x.shape[1] for x, _, _ in calls} == {7, 11, 13, 17}
    summary = json.loads(paths["summary"].read_text())
    assert summary["reference_max_abs_differences"] == {"oof": 0., "camera": 0., "submission": 0.}
    reference = summary["experiments"][0]
    assert reference["loo_emd"] == pytest.approx(100.)
    assert reference["camera_pairs"] == 1
    comparison = pd.read_csv(paths["comparison"])
    assert comparison["sample_id"].tolist() == ["B", "A", "D", "C"]
    np.testing.assert_allclose(comparison["emd_ridge_rgb_reference"], [200, 200, 0, 0])
    for name in ("ridge_rgb_texture_100", "ridge_gray_100", "ridge_gray_texture_100"):
        submission = pd.read_csv(paths[f"{name}_submission"])
        assert submission["sample_id"].tolist() == ["T2", "T1"]
        assert submission.shape == (2, 12)
    np.testing.assert_allclose(pd.read_csv(paths["ridge_gray_100_submission"])["0.002"], [50, 50])
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("corruption", ["duplicate", "camera", "nonfinite"])
def test_texture_run_rejects_inconsistent_feature_cache(texture_run, corruption):
    config, _artifacts, first = texture_run
    path = first / "gray_100_photo_features.csv"
    frame = pd.read_csv(path)
    if corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif corruption == "camera":
        frame.loc[0, "camera"] = "wrong camera"
    else:
        frame.loc[0, "feature_0"] = np.inf
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cache"):
        write_texture_experiment(config)


@pytest.mark.parametrize("source", ["oof_predictions.csv", "camera_predictions.csv"])
def test_texture_run_stops_when_reference_predictions_do_not_reconstruct(texture_run, source):
    config, artifacts, first = texture_run
    path = first / source
    frame = pd.read_csv(path)
    frame.loc[0, "0.002"] -= 1.
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="reference"):
        write_texture_experiment(config)
    assert not (artifacts / "submissions" / "ridge_rgb_texture_100.csv").exists()
