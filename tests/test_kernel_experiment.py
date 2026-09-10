import json

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.kernel_experiment import write_kernel_experiment


def test_kernel_experiment_aligns_soils_and_preserves_inputs(tmp_path) -> None:
    curated = tmp_path / "curated"
    artifacts = tmp_path / "artifacts"
    submissions = tmp_path / "candidates"
    first_batch = artifacts / "experiments" / "first_batch"
    multicrop = artifacts / "experiments" / "multicrop"
    for directory in (curated, first_batch, multicrop):
        directory.mkdir(parents=True)
    config = tmp_path / "data.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n"
        f"  submissions_dir: {submissions}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n"
    )

    def curves(ids, values):
        frame = pd.DataFrame({"sample_id": ids})
        frame[list(CANONICAL_GRAIN_LABELS)] = [[v] * 10 + [100.0] for v in values]
        return frame

    curves(["B", "A", "C"], [80.0, 20.0, 50.0]).to_csv(curated / "train.csv", index=False)
    template = curves(["T2", "T1"], [0.0, 0.0])
    template[list(CANONICAL_GRAIN_LABELS)] = 0.0
    template.to_csv(curated / "sample.csv", index=False)
    photos = pd.DataFrame({
        "sample_id": ["T1", "C", "A", "T2", "B"],
        "split": ["test", "train", "train", "test", "train"],
        "camera": ["phone"] * 5,
    })
    for i in range(13):
        photos[f"feature_{i}"] = [-1.0, 0.0, -1.0, 0.0, 1.0]
    photos.to_csv(first_batch / "reference_rgb_100_photo_features.csv", index=False)
    reference = curves(["C", "A", "B"], [40.0] * 3)
    reference.insert(0, "experiment", "ridge_rgb_100")
    reference["emd"] = 999.0  # Recompute errors, rather than trusting stored scores.
    reference.to_csv(first_batch / "oof_predictions.csv", index=False)
    curves(["C", "B", "A"], [60.0] * 3).to_csv(multicrop / "oof_predictions.csv", index=False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    paths = write_kernel_experiment(config)

    submission = pd.read_csv(paths["submission"])
    comparison = pd.read_csv(paths["comparison"])
    summary = json.loads(paths["summary"].read_text())
    assert paths["submission"] == submissions / "kernel_ridge_rgb_100.csv"
    assert list(submission.columns) == list(template.columns)
    assert submission["sample_id"].tolist() == ["T2", "T1"]
    assert submission.loc[0, "0.002"] == pytest.approx(50.0)
    contrast = 1.0 - np.exp(-6.0)
    assert submission.loc[1, "0.002"] == pytest.approx(50.0 - 30.0 * contrast / (1.0 + contrast))
    assert comparison["sample_id"].tolist() == ["B", "A", "C"]
    np.testing.assert_allclose(comparison["emd_ridge"], [200.0, 100.0, 50.0])
    np.testing.assert_allclose(comparison["emd_multicrop"], [100.0, 200.0, 50.0])
    assert summary["loo_emd"] == pytest.approx(comparison["emd_kernel"].mean())
    assert summary["camera_pairs"] == 0
    assert summary["paired_camera_disagreement_emd"] is None
    assert len(summary["sources"]) == 5
    assert all(len(source["sha256"]) == 64 for source in summary["sources"])
    assert all(p.read_bytes() == content for p, content in before.items())
