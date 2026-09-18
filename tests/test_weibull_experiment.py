import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
import yaml

import soilgrain.weibull_experiment as experiment
from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.experiments import evaluate_samples
from soilgrain.ridge import ridge_curves
from soilgrain.search_inputs import PINNED_INPUTS
from soilgrain.targets import curve_array


def pin_label_sources(config, monkeypatch):
    relative = "experiments/physical_photo/summary.json"
    manifest = config.parent / "artifacts" / relative
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"sources": [
        {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
        for path in (config, config.parent / "train.csv")
    ]}))
    monkeypatch.setitem(PINNED_INPUTS, relative, sha256(manifest.read_bytes()).hexdigest())


def fixture_inputs(tmp_path, monkeypatch, *, bad_capacity=False):
    ids = ["A", "B", "C", "D"]
    diameter = np.asarray(SUPPORT_DIAMETERS)
    values = np.array([100 * (1 - np.exp(-(diameter / scale) ** 0.7))
                       for scale in [0.1, 0.3, 1., 3.]])
    values[:, -1] = 100.
    if bad_capacity:
        values[:] = [0., 0., 40., 40., 40., 40., 40., 40., 90., 90., 100.]
    truth = pd.DataFrame(values, columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0, "sample_id", ids)
    train_path = tmp_path / "train.csv"
    truth.to_csv(train_path, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {
        "curated_dir": str(tmp_path), "artifacts_dir": str(tmp_path / "artifacts"),
    }, "files": {"train": "train.csv"}}))
    pin_label_sources(config, monkeypatch)
    sample = pd.DataFrame([["Q", *([0.] * 11)]], columns=["sample_id", *CANONICAL_GRAIN_LABELS])
    sample.rename(columns={"200.0": "200", "2.0": "2"}, inplace=True)
    rng = np.random.default_rng(3)
    photos = pd.DataFrame(rng.normal(size=(10, 23)), columns=[f"feature_{i}" for i in range(23)])
    photos["sample_id"] = np.repeat(ids + ["Q"], 2)
    photos["camera"] = ["Motorola Edge", "Samsung A52"] * 5
    photos["split"] = ["train"] * 8 + ["test"] * 2
    photos["path"] = [f"photo{i}.jpg" for i in range(10)]

    def load_inputs(_):
        verified_truth = pd.read_csv(train_path)
        oof, _ = evaluate_samples(photos.iloc[:8], ids, curve_array(verified_truth), ridge_curves)
        reference = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
        reference.insert(0, "sample_id", ids)
        return verified_truth, sample, {"spectral": photos}, {"spectral": {"oof": reference}}, []

    monkeypatch.setattr(experiment, "load_search_inputs", load_inputs)
    return config, truth, sample, photos


def test_failed_capacity_gate_writes_reconstruction_only(tmp_path, monkeypatch) -> None:
    config, truth, _, _ = fixture_inputs(tmp_path, monkeypatch, bad_capacity=True)

    def forbid_regression_inputs(_):
        raise AssertionError("Failed capacity gate must not access image features")

    monkeypatch.setattr(experiment, "load_search_inputs", forbid_regression_inputs)
    paths = experiment.run_weibull(config)
    summary = json.loads(paths["summary"].read_text())

    assert summary["capacity_gate_passed"] is False
    assert summary["reconstruction_max_emd"] > 15.
    assert summary["status"] == "stopped_at_capacity_gate"
    assert len(summary["producing_commit"]) == 40
    assert "submission" not in paths
    reconstructed = pd.read_csv(paths["reconstruction"])
    assert reconstructed.sample_id.tolist() == truth.sample_id.tolist()
    assert (reconstructed["200.0"] == 100.).all()


def test_predictions_exclude_heldout_labels_and_test_rows(tmp_path, monkeypatch) -> None:
    config, truth, sample, photos = fixture_inputs(tmp_path, monkeypatch)
    paths = experiment.run_weibull(config)
    first = pd.read_csv(paths["oof"])
    transfer = pd.read_csv(paths["camera_transfer"])
    summary = json.loads(paths["summary"].read_text())
    submission = pd.read_csv(paths["submission"])

    assert summary["capacity_gate_passed"] is True
    assert len(summary["producing_commit"]) == 40
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission["sample_id"].tolist() == ["Q"]
    assert len(transfer) == 16
    paired = pd.read_csv(paths["camera_transfer_pairs"])
    assert len(paired) == 8
    np.testing.assert_allclose(paired.transfer_minus_control_emd, paired.transfer_emd - paired.control_emd)
    assert all(row.sample_id not in json.loads(row.training_soil_ids) for row in transfer.itertuples())
    assert {path.parent.name for path in paths.values()} == {"weibull"}
    assert summary["loo_emd"] == first.emd.mean()
    assert "mean_improvement_excluding_largest_beneficiary" in summary

    photos.loc[photos.split == "test", [f"feature_{i}" for i in range(23)]] *= 1e9
    query_changed = experiment.run_weibull(config)
    np.testing.assert_allclose(curve_array(first), curve_array(pd.read_csv(query_changed["oof"])))
    np.testing.assert_allclose(curve_array(transfer), curve_array(pd.read_csv(query_changed["camera_transfer"])))

    diameter = np.asarray(SUPPORT_DIAMETERS)
    changed = 100 * (1 - np.exp(-(diameter / 10.) ** 1.2))
    changed[-1] = 100.
    truth.loc[0, list(CANONICAL_GRAIN_LABELS)] = changed
    truth.to_csv(config.parent / "train.csv", index=False)
    pin_label_sources(config, monkeypatch)
    repeated = experiment.run_weibull(config)
    second = pd.read_csv(repeated["oof"])
    second_transfer = pd.read_csv(repeated["camera_transfer"])

    np.testing.assert_allclose(curve_array(first.iloc[:1]), curve_array(second.iloc[:1]))
    np.testing.assert_allclose(curve_array(transfer[transfer.sample_id == "A"]),
                               curve_array(second_transfer[second_transfer.sample_id == "A"]))


@pytest.mark.parametrize("filename", ["train.csv", "config.yaml"])
def test_rejects_changed_labels_or_config_before_capacity_fit(tmp_path, monkeypatch, filename):
    config, _, _, _ = fixture_inputs(tmp_path, monkeypatch, bad_capacity=True)
    path = config.parent / filename
    path.write_text(path.read_text() + "\n")

    with pytest.raises(ValueError, match="Recorded source changed"):
        experiment.run_weibull(config)
    assert not (config.parent / "artifacts/experiments/weibull").exists()
