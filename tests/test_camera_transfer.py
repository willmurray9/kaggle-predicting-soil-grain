import json
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.pls import pls_curves
from soilgrain.ridge import ridge_curves


CAMERAS = ("Motorola Edge", "Samsung A52")


@pytest.fixture
def paired_data():
    ids = ["C", "A", "D", "B", "extra"]
    curves = np.array([np.r_[np.linspace(i * 4, 70 + i * 4, 10), 100.] for i in range(5)])
    rows = []
    for i, soil in enumerate(ids):
        for camera_index, camera in enumerate(CAMERAS if soil != "extra" else CAMERAS[:1]):
            for photo in range(i + 1):
                rows.append({"split": "train", "sample_id": soil, "camera": camera,
                             "path": f"{soil}_{camera_index}_{photo}.jpg",
                             "feature_0": float(i), "feature_1": float(i * i + camera_index * 3)})
    rows.append({"split": "test", "sample_id": "test", "camera": "iPhone 14", "path": "test.jpg",
                 "feature_0": 1e6, "feature_1": 1e6})
    return pd.DataFrame(rows), ids, curves


def evaluate(*args):
    from soilgrain.camera_transfer import evaluate_camera_transfer
    return evaluate_camera_transfer(*args)


def test_folds_exclude_both_views_and_pair_controls_with_identical_twenty_soils():
    ids = [f"soil_{i}" for i in range(21)]
    curves = np.tile(np.linspace(0, 100, 11), (21, 1))
    rows = [{"split": "train", "sample_id": soil, "camera": camera,
             "feature_0": i, "feature_1": 100 * k}
            for i, soil in enumerate(ids) for k, camera in enumerate(CAMERAS)]
    calls = []

    def predictor(train, targets, query):
        calls.append((train.copy(), targets.copy(), query.copy()))
        return ridge_curves(train, targets, query)

    result = evaluate(pd.DataFrame(rows), ids, curves, predictor)
    assert len(result) == 84
    assert len(calls) == 42
    for i, soil in enumerate(ids):
        fold = result[result.sample_id == soil]
        assert set(fold.direction) == {f"{a} -> {b}" for a in CAMERAS for b in CAMERAS}
        expected_ids = [value for value in ids if value != soil]
        assert fold.training_soil_ids.map(json.loads).tolist() == [expected_ids] * 4
        assert fold.training_soils.eq(20).all()
        for camera_index in range(2):
            train, targets, query = calls[i * 2 + camera_index]
            assert train.shape == (20, 2)
            assert set(train[:, 0]) == set(range(21)) - {i}
            assert np.all(train[:, 1] == 100 * camera_index)
            np.testing.assert_array_equal(query, [[i, 0], [i, 100]])
            assert targets.shape == (20, 11)


@pytest.mark.parametrize("predictor", [ridge_curves, partial(pls_curves, n_components=1)])
def test_heldout_label_and_test_rows_cannot_change_heldout_predictions(paired_data, predictor):
    photos, ids, curves = paired_data
    before = evaluate(photos, ids, curves, predictor)
    changed = curves.copy()
    changed[0] = np.linspace(50, 100, 11)
    photos.loc[photos.split == "test", ["feature_0", "feature_1"]] = [-1e9, 1e9]
    after = evaluate(photos, ids, changed, predictor)
    np.testing.assert_array_equal(before.loc[before.sample_id == "C", CANONICAL_GRAIN_LABELS],
                                  after.loc[after.sample_id == "C", CANONICAL_GRAIN_LABELS])
    assert set(before.sample_id) == {"A", "B", "C", "D"}


def test_query_changes_cannot_enter_training_or_change_other_query_control(paired_data):
    photos, ids, curves = paired_data
    before = evaluate(photos, ids, curves, ridge_curves)
    photos.loc[(photos.sample_id == "C") & (photos.camera == CAMERAS[1]), "feature_1"] += 5000
    after = evaluate(photos, ids, curves, ridge_curves)
    keep = (before.sample_id == "C") & (before.query_camera == CAMERAS[0])
    np.testing.assert_array_equal(before.loc[keep, CANONICAL_GRAIN_LABELS], after.loc[keep, CANONICAL_GRAIN_LABELS])


def test_unequal_photo_counts_have_equal_training_and_macro_score_weight(paired_data):
    from soilgrain.camera_transfer import summarize_camera_transfer
    photos, ids, curves = paired_data
    before = evaluate(photos, ids, curves, ridge_curves)
    photos = pd.concat([photos, *[photos.loc[photos.sample_id == "B"]] * 10], ignore_index=True)
    after = evaluate(photos, ids, curves, ridge_curves)
    np.testing.assert_array_equal(before[list(CANONICAL_GRAIN_LABELS)], after[list(CANONICAL_GRAIN_LABELS)])
    directions, paired = summarize_camera_transfer(after)
    assert len(directions) == 4
    assert len(paired) == 8
    for row in directions.itertuples():
        scores = after.loc[after.direction == row.direction, "emd"]
        assert row.macro_soil_emd == pytest.approx(scores.mean())
        assert row.soils == 4
    for row in paired.itertuples():
        assert row.transfer_minus_control_emd == pytest.approx(row.transfer_emd - row.control_emd)
        assert json.loads(row.training_soil_ids) == [soil for soil in ids[:4] if soil != row.sample_id]


def test_fine_threshold_contributions_sum_to_competition_emd(paired_data):
    photos, ids, curves = paired_data
    result = evaluate(photos, ids, curves, ridge_curves)
    first = result.iloc[0]
    widths = np.diff(np.log10(SUPPORT_DIAMETERS))
    errors = np.abs(curves[0, :10] - first[list(CANONICAL_GRAIN_LABELS[:10])].to_numpy(dtype=float))
    contributions = errors * widths
    np.testing.assert_allclose(first[[f"emd_contribution_{label}" for label in CANONICAL_GRAIN_LABELS[:10]]]
                               .to_numpy(dtype=float), contributions)
    assert first.emd == pytest.approx(contributions.sum())
    assert first.fine_emd == pytest.approx(contributions[:4].sum())


def test_invalid_or_unpaired_training_data_fails_explicitly(paired_data):
    photos, ids, curves = paired_data
    with pytest.raises(ValueError, match="paired"):
        evaluate(photos[photos.camera == CAMERAS[0]], ids, curves, ridge_curves)
    photos.loc[0, "feature_0"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate(photos, ids, curves, ridge_curves)


@pytest.fixture
def camera_run(tmp_path, monkeypatch):
    from soilgrain import camera_transfer
    artifacts = tmp_path / "artifacts"
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  artifacts_dir: {artifacts}\n")
    ids = [f"soil_{i}" for i in range(24)]
    truth = pd.DataFrame([np.r_[np.linspace(i, 70 + i, 10), 100.] for i in range(24)],
                         columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0, "sample_id", ids)
    rows = [{"split": "train", "sample_id": soil, "camera": camera, "path": f"{soil}_{k}.jpg"}
            for i, soil in enumerate(ids) for k, camera in enumerate(CAMERAS if i < 21 else ("extra",))]
    rows.append({"split": "test", "sample_id": "test", "camera": "iPhone 14", "path": "test.jpg"})
    index = pd.DataFrame(rows)
    rng = np.random.default_rng(8)
    frames = {}
    for name, count in [("spectral", 23), ("dino", 384)]:
        dtype = np.float32 if name == "dino" else np.float64
        values = pd.DataFrame(rng.normal(size=(len(index), count)).astype(dtype),
                              columns=[f"feature_{i}" for i in range(count)])
        frames[name] = pd.concat([index, values], axis=1)
    first = artifacts / "experiments/first_batch"
    dino = artifacts / "experiments/autonomous_search/dino_pls"
    first.mkdir(parents=True)
    dino.mkdir(parents=True)
    rgb_path = first / "reference_rgb_100_photo_features.csv"
    frames["spectral"][list(index) + [f"feature_{i}" for i in range(13)]].iloc[::-1].to_csv(rgb_path, index=False)
    oof = truth.copy()
    oof.insert(0, "experiment", "ridge_rgb_100")
    oof.to_csv(first / "oof_predictions.csv", index=False)
    truth.to_csv(dino / "oof.csv", index=False)
    pd.DataFrame({"sample_id": ids, "n_components": 1, "selected": True}).to_csv(dino / "outer_selection.csv", index=False)
    sources = [{"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
               for path in [config, rgb_path, first / "oof_predictions.csv"]]
    monkeypatch.setattr(camera_transfer, "load_search_inputs", lambda _: (
        truth, pd.DataFrame({"sample_id": ["test"]}), frames, {"spectral": {"oof": truth.copy()}}, sources.copy(),
    ), raising=False)
    monkeypatch.setattr(camera_transfer, "PINNED_POOLED_DINO", {
        str(path.relative_to(artifacts)): sha256(path.read_bytes()).hexdigest()
        for path in [dino / "oof.csv", dino / "outer_selection.csv"]
    }, raising=False)
    return config, artifacts, frames, rgb_path, sources


def test_runner_preserves_sources_saves_all_folds_and_separates_pooled_reference(camera_run):
    from soilgrain.camera_transfer import run_camera_transfer
    config, artifacts, _, _, _ = camera_run
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    summary = run_camera_transfer(config)
    assert summary["paired_soils"] == 21
    assert summary["training_soils_per_fold"] == 20
    assert len(summary["directions"]) == 12
    assert len(summary["paired_transfer_minus_control"]) == 6
    assert all(row["soils"] == 24 and row["macro_soil_emd"] == pytest.approx(0, abs=1e-12)
               for row in summary["existing_pooled_loo"])
    predictions = pd.read_csv(summary["outputs"]["predictions"]["path"])
    assert len(predictions) == 252
    assert len(pd.read_csv(summary["outputs"]["paired_comparison"]["path"])) == 126
    assert all(len(json.loads(ids)) == 20 for ids in predictions.training_soil_ids)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert json.loads((artifacts / "experiments/camera_transfer/summary.json").read_text()) == summary
    for item in [*summary["sources"], *summary["outputs"].values()]:
        assert sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_runner_rejects_wrong_rgb_mapping_even_when_source_hash_is_current(camera_run):
    from soilgrain.camera_transfer import run_camera_transfer
    config, artifacts, frames, _, _ = camera_run
    frames["spectral"].loc[0, "feature_0"] += 1
    with pytest.raises(ValueError, match="RGB.*mapping"):
        run_camera_transfer(config)
    assert not (artifacts / "experiments/camera_transfer").exists()


def test_runner_rejects_unverified_original_rgb_cache(camera_run):
    from soilgrain.camera_transfer import run_camera_transfer
    config, artifacts, _, rgb_path, sources = camera_run
    sources[:] = [item for item in sources if item["path"] != str(rgb_path)]
    with pytest.raises(ValueError, match="verified"):
        run_camera_transfer(config)
    assert not (artifacts / "experiments/camera_transfer").exists()
