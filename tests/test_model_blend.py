import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.model_blend import _blend_predictions, write_model_blend


def _frame(ids, values, cameras=None):
    frame = pd.DataFrame({"sample_id": ids})
    if cameras is not None:
        frame["camera"] = cameras
    frame[list(CANONICAL_GRAIN_LABELS)] = [[value] * 10 + [100.0] for value in values]
    return frame


def test_blend_aligns_soil_and_camera_together():
    left = _frame(["A", "A", "B"], [10, 20, 30], ["left", "right", "left"])
    right = _frame(["B", "A", "A"], [70, 80, 50], ["left", "right", "left"])

    result = _blend_predictions(left, right, ["sample_id", "camera"])

    np.testing.assert_allclose(result["0.002"], [30, 50, 50])
    assert result[["sample_id", "camera"]].equals(left[["sample_id", "camera"]])


@pytest.mark.parametrize("bad", [
    _frame(["A"], [10]),
    _frame(["A", "A"], [10, 20]),
    _frame(["A", None], [10, 20]),
    _frame(["A", "B"], [-1, 20]),
    _frame(["A", "B"], [10, 20]).iloc[:0],
], ids=["coverage", "duplicate", "null", "invalid-curve", "empty"])
def test_blend_rejects_invalid_components(bad):
    with pytest.raises(ValueError):
        _blend_predictions(_frame(["A", "B"], [20, 30]), bad, ["sample_id"])


@pytest.fixture
def blend_files(tmp_path):
    curated = tmp_path / "curated"
    experiments = tmp_path / "artifacts" / "experiments"
    candidates = tmp_path / "candidates"
    for directory in (curated, experiments / "multicrop", experiments / "nested_ridge", candidates):
        directory.mkdir(parents=True)
    config = tmp_path / "data.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {tmp_path / 'artifacts'}\n"
        f"  submissions_dir: {candidates}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n"
    )
    _frame(["B", "A"], [20, 50]).to_csv(curated / "train.csv", index=False)
    sample = _frame(["T2", "T1"], [0, 0]).rename(columns={"2.0": "2", "200.0": "200"})
    sample.iloc[:, 1:] = 0.0
    sample.to_csv(curated / "sample.csv", index=False)
    for name, oof, cameras, candidate in (
        ("multicrop", _frame(["A", "B"], [20, 10]),
         _frame(["A", "A", "B"], [20, 80, 0], ["left", "right", "left"]),
         _frame(["T1", "T2"], [40, 10])),
        ("nested_ridge", _frame(["B", "A"], [50, 80]),
         _frame(["B", "A", "A"], [40, 20, 80], ["left", "right", "left"]),
         _frame(["T2", "T1"], [50, 80])),
    ):
        oof["emd"] = 999.0
        cameras["emd"] = 999.0
        oof.to_csv(experiments / name / "oof_predictions.csv", index=False)
        cameras.to_csv(experiments / name / "camera_predictions.csv", index=False)
        (experiments / name / "summary.json").write_text('{"loo_emd": 999.0}')
        filename = "multicrop_baseline.csv" if name == "multicrop" else "frozen_resnet18_nested_ridge.csv"
        candidate.columns = sample.columns
        candidate.to_csv(candidates / filename, index=False)
    return config


def test_writer_scores_averaged_curves_and_preserves_sources(blend_files):
    config = blend_files
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}

    paths = write_model_blend(config)

    oof = pd.read_csv(paths["oof"])
    cameras = pd.read_csv(paths["camera_predictions"])
    comparison = pd.read_csv(paths["comparison"])
    submission = pd.read_csv(paths["submission"])
    summary = json.loads(paths["summary"].read_text())
    assert oof["sample_id"].tolist() == ["B", "A"]
    np.testing.assert_allclose(oof["emd"], [50, 0])
    np.testing.assert_allclose(cameras["emd"], [0, 0, 0])
    np.testing.assert_allclose(comparison["emd_multicrop"], [50, 150])
    np.testing.assert_allclose(comparison["emd_nested_ridge"], [150, 150])
    np.testing.assert_allclose(comparison["improvement_vs_multicrop"], [0, 150])
    assert summary["loo_emd"] == pytest.approx(25)
    assert summary["paired_camera_disagreement_emd"] == pytest.approx(0)
    assert summary["camera_pairs"] == 1
    assert summary["train_samples"] == summary["test_samples"] == 2
    assert summary["weights"] == {"multicrop": 0.5, "nested_ridge": 0.5}
    assert paths["submission"].name == "multicrop_nested_ridge_blend.csv"
    assert submission["sample_id"].tolist() == ["T2", "T1"]
    np.testing.assert_allclose(submission["0.002"], [30, 60])
    assert submission.columns.tolist() == pd.read_csv(config.parent / "curated" / "sample.csv").columns.tolist()
    assert len(summary["sources"]) == len(before) == 11
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all(source["sha256"] == sha256(before[Path(source["path"])]).hexdigest()
               for source in summary["sources"])


@pytest.mark.parametrize("filename", ["oof_predictions.csv", "camera_predictions.csv"])
def test_writer_rejects_jointly_missing_labeled_soil(blend_files, filename):
    experiments = blend_files.parent / "artifacts" / "experiments"
    for name in ("multicrop", "nested_ridge"):
        path = experiments / name / filename
        frame = pd.read_csv(path)
        frame[frame["sample_id"] != "B"].to_csv(path, index=False)

    with pytest.raises(ValueError, match="coverage"):
        write_model_blend(blend_files)
    assert not (experiments / "model_blend").exists()


def test_writer_rejects_mismatched_camera_keys(blend_files):
    experiments = blend_files.parent / "artifacts" / "experiments"
    path = experiments / "nested_ridge" / "camera_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "camera"] = "different_camera"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="coverage"):
        write_model_blend(blend_files)
    assert not (experiments / "model_blend").exists()


def test_writer_requires_submission_template_coverage(blend_files):
    path = blend_files.parent / "candidates" / "frozen_resnet18_nested_ridge.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "sample_id"] = "unexpected_soil"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="sample_id mismatch"):
        write_model_blend(blend_files)
    assert not (blend_files.parent / "artifacts" / "experiments" / "model_blend").exists()
