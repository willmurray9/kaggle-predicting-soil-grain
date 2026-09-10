import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.spatial_experiment import write_spatial_experiment


def test_spatial_experiment_excludes_whole_soils_and_aligns_outputs(tmp_path):
    curated, artifacts = tmp_path / "curated", tmp_path / "artifacts"
    reports = artifacts / "reports"
    reference_dir = artifacts / "experiments" / "multicrop"
    for directory in (curated, reports, reference_dir):
        directory.mkdir(parents=True)
    config = tmp_path / "data.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n"
        f"  reports_dir: {reports}\n  submissions_dir: {artifacts / 'submissions'}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n  ppm: ppm.csv\n"
    )

    def curves(ids, values):
        frame = pd.DataFrame({"sample_id": ids})
        frame[list(CANONICAL_GRAIN_LABELS)] = [[v] * 10 + [100.] for v in values]
        return frame

    curves(["D", "B", "A", "C"], [90, 30, 0, 60]).to_csv(curated / "train.csv", index=False)
    template = curves(["T2", "T1"], [0, 0])
    template.to_csv(curated / "sample.csv", index=False)
    pd.DataFrame({"phone": ["one", "two"], "width": [256, 256], "height": [256, 256],
                  "ppm": [1., 1.]}).to_csv(curated / "ppm.csv", index=False)
    rows = []
    # A has unequal photo counts per camera. Its photo mean must be 40/255,
    # whereas equal camera means would be 30/255.
    for soil, camera, brightness in [("A", "one", 0), ("A", "two", 60), ("A", "two", 60),
                                     ("B", "one", 80), ("C", "one", 120), ("D", "one", 160),
                                     ("T1", "one", 40), ("T2", "one", 160)]:
        path = tmp_path / f"photo_{len(rows)}.png"
        Image.new("RGB", (256, 256), (brightness,) * 3).save(path)
        rows.append({"split": "test" if soil.startswith("T") else "train",
                     "sample_id": soil, "camera": camera, "path": str(path)})
    index_path = reports / "photo_index.csv"
    pd.DataFrame(rows[::-1]).to_csv(index_path, index=False)
    curves(["C", "A", "B", "D"], [0, 0, 0, 0]).to_csv(reference_dir / "oof_predictions.csv", index=False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    paths = write_spatial_experiment(config)

    oof = pd.read_csv(paths["oof"])
    blend = oof[oof["experiment"] == "spatial_multicrop"]
    assert blend["sample_id"].tolist() == ["D", "B", "A", "C"]
    # Three training neighbors are exactly the other three physical soils.
    np.testing.assert_allclose(blend["0.002"], [30, 50, 60, 40])
    cameras = pd.read_csv(paths["camera_predictions"])
    views = cameras[(cameras["experiment"] == "spatial_multicrop") & (cameras["sample_id"] == "A")]
    np.testing.assert_allclose(views["0.002"], [60, 60])
    submission = pd.read_csv(paths["submission"])
    assert list(submission.columns) == list(template.columns)
    assert submission["sample_id"].tolist() == ["T2", "T1"]
    np.testing.assert_allclose(submission["0.002"], [60, 30])
    comparison = pd.read_csv(paths["comparison"])
    np.testing.assert_allclose(comparison["emd_reference"], [450, 150, 0, 300])
    np.testing.assert_allclose(comparison["emd_spatial"], [300, 100, 300, 100])
    features = pd.read_csv(paths["gray_50_features"])
    assert len(features) == 8
    assert features.loc[features["sample_id"] == "A", "feature_0"].mean() == pytest.approx(40 / 255)
    summary = json.loads(paths["summary"].read_text())
    assert summary["experiments"][-1]["loo_emd"] == pytest.approx(200)
    assert summary["experiments"][-1]["camera_pairs"] == 1
    assert all(p.read_bytes() == content for p, content in before.items())

    pd.concat([pd.DataFrame(rows), pd.DataFrame(rows[:1])]).to_csv(index_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        write_spatial_experiment(config)
