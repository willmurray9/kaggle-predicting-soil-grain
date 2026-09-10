import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.nested_experiment import write_nested_experiment


def test_nested_experiment_uses_cache_and_keeps_test_data_out_of_selection(tmp_path) -> None:
    curated = tmp_path / "curated"
    artifacts = tmp_path / "artifacts"
    frozen = artifacts / "experiments" / "frozen_resnet18"
    multicrop = artifacts / "experiments" / "multicrop"
    for directory in (curated, frozen, multicrop):
        directory.mkdir(parents=True)
    config = tmp_path / "data.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n"
        f"  submissions_dir: {tmp_path / 'candidates'}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n"
    )

    def curves(ids, values):
        frame = pd.DataFrame({"sample_id": ids})
        frame[list(CANONICAL_GRAIN_LABELS)] = [[v] * 10 + [100.0] for v in values]
        return frame

    curves(["B", "A", "C"], [80.0, 20.0, 50.0]).to_csv(curated / "train.csv", index=False)
    sample = curves(["T2", "T1"], [0.0, 0.0])
    sample[list(CANONICAL_GRAIN_LABELS)] = 0.0
    sample.to_csv(curated / "sample.csv", index=False)
    photos = pd.DataFrame({
        "sample_id": ["T1", "C", "A", "T2", "B", "A"],
        "split": ["test", "train", "train", "test", "train", "train"],
        "camera": ["phone"] * 5 + ["other_phone"],
        "path": [f"unused_photo_{i}.jpg" for i in range(6)],
    })
    columns = [f"feature_{i}" for i in range(512)]
    photos = pd.concat([photos, pd.DataFrame(
        np.tile(np.array([-1.0, 0.0, -1.0, 0.0, 1.0, -1.0])[:, None], (1, 512)),
        columns=columns,
    )], axis=1)
    cache = frozen / "photo_features.csv"
    photos.to_csv(cache, index=False)
    (frozen / "summary.json").write_text(json.dumps({"encoder": {"checkpoint_sha256": "a" * 64}}))
    reference = curves(["C", "A", "B"], [40.0] * 3)
    reference["emd"] = 999.0
    reference.to_csv(frozen / "oof_predictions.csv", index=False)
    curves(["C", "B", "A"], [60.0] * 3).to_csv(multicrop / "oof_predictions.csv", index=False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    paths = write_nested_experiment(config)

    submission = pd.read_csv(paths["submission"])
    comparison = pd.read_csv(paths["comparison"])
    summary = json.loads(paths["summary"].read_text())
    assert paths["submission"] == tmp_path / "candidates" / "frozen_resnet18_nested_ridge.csv"
    assert list(submission.columns) == list(sample.columns)
    assert submission["sample_id"].tolist() == ["T2", "T1"]
    assert submission.loc[0, "0.002"] == pytest.approx(50.0)
    assert comparison["sample_id"].tolist() == ["B", "A", "C"]
    np.testing.assert_allclose(comparison["emd_frozen"], [200.0, 100.0, 50.0])
    np.testing.assert_allclose(comparison["emd_multicrop"], [100.0, 200.0, 50.0])
    assert summary["outer_loo_emd"] == pytest.approx(comparison["emd_nested"].mean())
    assert summary["encoder"]["checkpoint_sha256"] == "a" * 64
    assert summary["camera_pairs"] == 1
    assert summary["paired_camera_disagreement_emd"] == pytest.approx(0.0)
    assert len(pd.read_csv(paths["outer_selection"])) == 9
    assert len(pd.read_csv(paths["final_selection"])) == 3
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all(source["sha256"] == sha256(before[Path(source["path"])]).hexdigest()
               for source in summary["sources"])

    validation_before = {key: paths[key].read_bytes() for key in (
        "oof", "camera_predictions", "outer_selection", "final_selection", "comparison",
    )}
    photos.loc[photos["split"] == "test", columns] = 1e6
    photos.to_csv(cache, index=False)
    write_nested_experiment(config)
    assert all(paths[key].read_bytes() == content for key, content in validation_before.items())
    updated = json.loads(paths["summary"].read_text())
    assert updated["final_alpha"] == summary["final_alpha"]
    assert not pd.read_csv(paths["submission"]).equals(submission)
