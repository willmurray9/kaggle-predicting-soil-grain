import json

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.pca_experiment import submission_evidence, write_pca_experiment


@pytest.mark.parametrize("improvements,camera,eligible", [
    ([2.] * 24, 20., True),
    ([0.5] * 24, 20., False),
    ([6.] * 10 + [-1.] * 14, 20., False),
    ([2.] * 24, 30., False),
    ([1.] * 12 + [-2.] * 11 + [50.], 20., False),
    ([2.] * 24, None, False),
])
def test_submission_screen_requires_all_declared_evidence(improvements, camera, eligible):
    reference = np.full(24, 100.)
    result = submission_evidence(reference - improvements, reference, camera, 29.)
    assert result["eligible"] is eligible
    assert result["improved_soils"] == sum(value > 1e-9 for value in improvements)


def test_submission_sensitivity_does_not_remove_a_soil_from_the_primary_score():
    errors = np.array([98.] * 23 + [50.])
    reference = np.full(24, 100.)
    result = submission_evidence(errors, reference, 20., 29.)
    assert result["mean_improvement_emd"] == pytest.approx(4.)
    assert result["improvement_without_largest_emd"] == pytest.approx(2.)
    np.testing.assert_array_equal(errors, [98.] * 23 + [50.])


def test_submission_evidence_rejects_unpaired_errors_and_nonfinite_camera_scores():
    with pytest.raises(ValueError, match="pairs"):
        submission_evidence([1., 2.], [3.], 20., 29.)
    assert not submission_evidence([1., 1.], [3., 3.], 20., np.inf)["eligible"]


@pytest.fixture
def pca_run(tmp_path):
    curated, artifacts = tmp_path / "curated", tmp_path / "artifacts"
    frozen = artifacts / "experiments" / "frozen_resnet18"
    nested = artifacts / "experiments" / "nested_ridge"
    texture = artifacts / "experiments" / "physical_texture"
    submissions = artifacts / "submissions"
    for directory in (curated, frozen, nested, texture, submissions):
        directory.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n  submissions_dir: {submissions}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n"
    )
    ids = [f"S{i:02}" for i in range(12)][::-1]

    def curves(soil_ids, value):
        frame = pd.DataFrame({"sample_id": soil_ids})
        frame[list(CANONICAL_GRAIN_LABELS)] = [[value] * 10 + [100.]] * len(soil_ids)
        return frame

    curves(ids, 50.).to_csv(curated / "train.csv", index=False)
    curves(["T2", "T1"], 0.).to_csv(curated / "sample.csv", index=False)
    photos = pd.DataFrame([
        {"sample_id": soil, "split": "test" if soil.startswith("T") else "train",
         "camera": camera, "path": f"unused_{soil}_{camera}.jpg"}
        for soil in [*ids, "T1", "T2"] for camera in ("one", "two")
    ])
    features = np.random.default_rng(41).normal(size=(len(photos), 512)).astype(np.float32)
    photos = pd.concat([photos, pd.DataFrame(features, columns=[f"feature_{i}" for i in range(512)])], axis=1)
    photos.to_csv(frozen / "photo_features.csv", index=False)
    (frozen / "summary.json").write_text(json.dumps({"encoder": {"feature_count": 512}}))
    reference = curves(ids[::-1], 50.)
    reference.to_csv(nested / "oof_predictions.csv", index=False)
    views = curves([soil for soil in ids[::-1] for _ in range(2)], 50.)
    views.insert(1, "camera", ["two", "one"] * len(ids))
    views.to_csv(nested / "camera_predictions.csv", index=False)
    pd.DataFrame([{"sample_id": soil, "alpha": a, "inner_loo_emd": 0., "selected": a == 1000.}
                  for soil in ids[::-1] for a in (1000., 100., 10.)]).to_csv(nested / "outer_selection.csv", index=False)
    pd.DataFrame([{"alpha": a, "loo_selection_emd": 0., "selected": a == 1000.}
                  for a in (1000., 100., 10.)]).to_csv(nested / "final_selection.csv", index=False)
    (nested / "summary.json").write_text(json.dumps({"final_alpha": 1000.}))
    curves(["T1", "T2"], 50.).to_csv(submissions / "frozen_resnet18_nested_ridge.csv", index=False)
    # A worse incumbent makes both candidates eligible. Standalone PCA has
    # lower error; the reference's stored error is untrusted.
    reference[list(CANONICAL_GRAIN_LABELS)[:-1]] = 45.
    reference.insert(0, "experiment", "ridge_rgb_texture_100")
    reference["emd"] = 999.
    reference.to_csv(texture / "oof_predictions.csv", index=False)
    views[list(CANONICAL_GRAIN_LABELS)[:-1]] = 45.
    views.insert(0, "experiment", "ridge_rgb_texture_100")
    views.to_csv(texture / "camera_predictions.csv", index=False)
    (texture / "summary.json").write_text("{}")
    curves(["T1", "T2"], 45.).to_csv(submissions / "ridge_rgb_texture_100.csv", index=False)
    return config, artifacts, frozen, nested, texture


def test_pca_run_reconstructs_reference_aligns_blend_and_keeps_test_out_of_selection(pca_run):
    config, artifacts, frozen, _nested, _texture = pca_run
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}

    paths = write_pca_experiment(config)

    summary = json.loads(paths["summary"].read_text())
    assert summary["final_alpha"] == 1000.
    assert summary["selected_submission"] == str(paths["pca_submission"])
    assert summary["reference_max_abs_differences"] == {"oof": 0., "camera": 0., "submission": 0.}
    for key, value in (("pca_submission", 50.), ("blend_submission", 47.5)):
        submission = pd.read_csv(paths[key])
        assert submission["sample_id"].tolist() == ["T2", "T1"]
        np.testing.assert_allclose(submission["0.002"], value)
    comparison = pd.read_csv(paths["comparison"])
    assert comparison["sample_id"].tolist() == [f"S{i:02}" for i in range(12)][::-1]
    np.testing.assert_allclose(comparison["emd_incumbent"], 25.)
    np.testing.assert_allclose(comparison["emd_pca"], 0.)
    np.testing.assert_allclose(comparison["emd_blend"], 12.5)
    assert len(pd.read_csv(paths["outer_selection"])) == 36
    assert all(p.read_bytes() == content for p, content in before.items())
    retained = {key: paths[key].read_bytes() for key in ("oof", "camera_predictions", "outer_selection", "final_selection", "comparison")}
    cache = frozen / "photo_features.csv"
    photos = pd.read_csv(cache)
    photos.loc[photos["split"] == "test", [c for c in photos if c.startswith("feature_")]] = 1e6
    photos.to_csv(cache, index=False)
    write_pca_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())


def test_pca_run_rejects_different_reference_predictions_before_writing_candidates(pca_run):
    config, artifacts, _frozen, nested, _texture = pca_run
    path = nested / "oof_predictions.csv"
    reference = pd.read_csv(path)
    reference.loc[0, "0.002"] -= 1.
    reference.to_csv(path, index=False)
    with pytest.raises(ValueError, match="reference"):
        write_pca_experiment(config)
    assert not (artifacts / "submissions" / "frozen_resnet18_pca8_nested_ridge.csv").exists()
