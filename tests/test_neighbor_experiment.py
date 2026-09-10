import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.multicrop import COMPONENTS
from soilgrain.neighbor_experiment import write_neighbor_experiment


def test_neighbor_experiment_aligns_ids_preserves_sources_and_excludes_test_data(tmp_path) -> None:
    curated = tmp_path / "curated"
    artifacts = tmp_path / "artifacts"
    first = artifacts / "experiments" / "first_batch"
    reference = artifacts / "experiments" / "multicrop"
    for directory in (curated, first, reference):
        directory.mkdir(parents=True)
    config = tmp_path / "data.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n"
        f"  submissions_dir: {tmp_path / 'candidates'}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n"
    )
    ids = ["B", "A", "I", "E", "C", "H", "D", "G", "F"]
    values = np.linspace(10.0, 90.0, 9)
    labels = pd.DataFrame({"sample_id": ids})
    labels[list(CANONICAL_GRAIN_LABELS)] = [[v] * 10 + [100.0] for v in values]
    labels.to_csv(curated / "train.csv", index=False)
    sample = labels.iloc[:2].copy()
    sample["sample_id"] = ["T2", "T1"]
    sample[list(CANONICAL_GRAIN_LABELS)] = 0.0
    sample.to_csv(curated / "sample.csv", index=False)
    old = labels.iloc[::-1].copy()
    old[list(CANONICAL_GRAIN_LABELS)] = [40.0] * 10 + [100.0]
    old["emd"] = 999.0
    old.to_csv(reference / "oof_predictions.csv", index=False)
    (first / "manifest.json").write_text(json.dumps({"validation": "Grouped fixture"}))

    rows = [
        {"sample_id": soil, "split": "train", "camera": camera, "path": f"{soil}_{camera}.jpg",
         **{f"feature_{j}": i - 4 + offset for j in range(7)}}
        for i, soil in enumerate(ids)
        for camera, offset in (("left", -.25), ("right", .25))
    ]
    rows.extend({
        "sample_id": soil, "split": "test", "camera": "test_phone", "path": f"{soil}.jpg",
        **{f"feature_{j}": v for j in range(7)},
    } for soil, v in (("T1", -4.0), ("T2", 0.0)))
    caches = [first / f"{name}_photo_features.csv" for name in COMPONENTS]
    for cache in caches:
        pd.DataFrame(rows).iloc[::-1].to_csv(cache, index=False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    paths = write_neighbor_experiment(config)

    summary = json.loads(paths["summary"].read_text())
    submission = pd.read_csv(paths["submission"])
    comparison = pd.read_csv(paths["comparison"])
    assert submission["sample_id"].tolist() == ["T2", "T1"]
    assert list(submission.columns) == list(sample.columns)
    assert submission.loc[0, "0.002"] == pytest.approx(50.0)
    assert comparison["sample_id"].tolist() == ids
    np.testing.assert_allclose(comparison["emd_fixed3"], np.abs(values - 40.0) * 5.0)
    assert summary["outer_loo_emd"] == pytest.approx(comparison["emd_nested"].mean())
    assert summary["camera_pairs"] == 9
    assert len(pd.read_csv(paths["outer_selection"])) == 36
    assert len(pd.read_csv(paths["final_selection"])) == 4
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all(source["sha256"] == sha256(before[Path(source["path"])]).hexdigest()
               for source in summary["sources"])
    validation_before = {key: paths[key].read_bytes() for key in (
        "oof", "camera_predictions", "outer_selection", "final_selection", "comparison",
    )}

    for cache in caches:
        photos = pd.read_csv(cache)
        photos.loc[photos["split"] == "test", [f"feature_{i}" for i in range(7)]] = 1e6
        photos.to_csv(cache, index=False)
    write_neighbor_experiment(config)
    assert all(paths[key].read_bytes() == content for key, content in validation_before.items())
    assert json.loads(paths["summary"].read_text())["final_n_neighbors"] == summary["final_n_neighbors"]
    assert not pd.read_csv(paths["submission"]).equals(submission)
