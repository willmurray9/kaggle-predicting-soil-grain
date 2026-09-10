import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.frozen_experiment import write_frozen_experiment


def test_frozen_experiment_aligns_soils_and_records_provenance(tmp_path, monkeypatch) -> None:
    curated = tmp_path / "curated"
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    first_batch = artifacts / "experiments" / "first_batch"
    multicrop = artifacts / "experiments" / "multicrop"
    for directory in (curated, reports, first_batch, multicrop):
        directory.mkdir(parents=True)
    config = tmp_path / "data.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n"
        f"  reports_dir: {reports}\n  submissions_dir: {tmp_path / 'candidates'}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n  ppm: ppm.csv\n"
    )

    def curves(ids, values):
        frame = pd.DataFrame({"sample_id": ids})
        frame[list(CANONICAL_GRAIN_LABELS)] = [[v] * 10 + [100.0] for v in values]
        return frame

    curves(["B", "A", "C"], [80.0, 20.0, 50.0]).to_csv(curated / "train.csv", index=False)
    sample = curves(["T2", "T1"], [0.0, 0.0])
    sample[list(CANONICAL_GRAIN_LABELS)] = 0.0
    sample.to_csv(curated / "sample.csv", index=False)
    pd.DataFrame({"phone": ["phone"], "ppm": [2.0]}).to_csv(curated / "ppm.csv", index=False)
    ids = ["T1", "C", "A", "T2", "B"]
    photos = pd.DataFrame({
        "sample_id": ids, "split": ["test", "train", "train", "test", "train"],
        "camera": ["phone"] * 5, "path": [str(tmp_path / f"{i}.jpg") for i in ids],
    })
    for path in photos["path"]:
        Path(path).write_bytes(b"fixture photograph")
    photos.to_csv(reports / "photo_index.csv", index=False)
    reference = curves(["C", "A", "B"], [40.0] * 3)
    reference.insert(0, "experiment", "ridge_rgb_100")
    reference["emd"] = 999.0
    reference.to_csv(first_batch / "oof_predictions.csv", index=False)
    curves(["C", "B", "A"], [60.0] * 3).to_csv(multicrop / "oof_predictions.csv", index=False)

    def fake_extract(index, ppm):
        pd.testing.assert_frame_equal(index, photos)
        assert ppm["phone"].tolist() == ["phone"]
        features = pd.DataFrame(
            np.tile(np.array([-1.0, 0.0, -1.0, 0.0, 1.0])[:, None], (1, 512)),
            columns=[f"feature_{i}" for i in range(512)],
        )
        return pd.concat([index, features], axis=1), {"checkpoint_sha256": "a" * 64, "device": "cpu"}

    monkeypatch.setattr("soilgrain.frozen_experiment.extract_frozen_features", fake_extract)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    paths = write_frozen_experiment(config)

    submission = pd.read_csv(paths["submission"])
    comparison = pd.read_csv(paths["comparison"])
    summary = json.loads(paths["summary"].read_text())
    assert paths["submission"] == tmp_path / "candidates" / "frozen_resnet18_ridge.csv"
    assert list(submission.columns) == list(sample.columns)
    assert submission["sample_id"].tolist() == ["T2", "T1"]
    np.testing.assert_allclose(submission["0.002"], [50.0, 50.0 - 30.0 * 1536.0 / 1546.0])
    assert comparison["sample_id"].tolist() == ["B", "A", "C"]
    np.testing.assert_allclose(comparison["emd_ridge"], [200.0, 100.0, 50.0])
    np.testing.assert_allclose(comparison["emd_multicrop"], [100.0, 200.0, 50.0])
    assert summary["loo_emd"] == pytest.approx(comparison["emd_frozen"].mean())
    assert summary["encoder"]["checkpoint_sha256"] == "a" * 64
    assert summary["feature_count"] == 512
    assert summary["camera_pairs"] == 0
    assert summary["paired_camera_disagreement_emd"] is None
    assert len(summary["photo_sources"]) == 5
    assert all(len(source["sha256"]) == 64 for source in summary["photo_sources"])
    assert all(p.read_bytes() == content for p, content in before.items())
