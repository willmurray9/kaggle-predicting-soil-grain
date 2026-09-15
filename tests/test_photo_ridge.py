import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.photo_ridge import evaluate_photo_ridge, photo_ridge_curves
from soilgrain.ridge import ridge_curves


def _curves(values):
    return np.array([[value] * 10 + [100.0] for value in values])


def _photos(groups):
    return pd.DataFrame([
        {"sample_id": sample_id, "camera": f"camera_{i % 2}", "feature_0": value}
        for sample_id, values in groups.items()
        for i, value in enumerate(values)
    ])


def test_one_photo_per_soil_matches_existing_ridge():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(5, 9))
    features[:, -1] = 3.0
    ids = ["C", "A", "E", "D", "B"]
    photos = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(9)])
    photos.insert(0, "sample_id", ids)
    curves = np.column_stack([np.sort(rng.uniform(0, 100, (5, 10)), axis=1), np.full(5, 100.0)])
    query = rng.normal(size=(4, 9))

    actual = photo_ridge_curves(photos.iloc[::-1], ids, curves, query)

    np.testing.assert_allclose(actual, ridge_curves(features, curves, query), atol=1e-12)


def test_unequal_photo_counts_match_analytic_soil_weighted_solution():
    photos = _photos({"A": [-1.0], "B": [-1.0, 1.0], "C": [-1.0, 1.0, 3.0]})

    predictions = photo_ridge_curves(photos, ["A", "B", "C"], _curves([20.0, 50.0, 80.0]), [[1.0]])

    # Soil means are -1, 0, 1. Weighted raw X'X is 17/3, X'y is 60,
    # and standardization turns alpha 10 into raw slope penalty 20/3.
    np.testing.assert_allclose(predictions[0, :10], 50.0 + 180.0 / 37.0)
    assert predictions[0, -1] == 100.0


def test_duplicating_all_views_within_one_soil_does_not_change_its_influence():
    photos = _photos({"A": [-1.0], "B": [-1.0, 1.0], "C": [-1.0, 1.0, 3.0]})
    repeated = pd.concat([photos, photos[photos.sample_id == "C"]], ignore_index=True)
    curves = _curves([20.0, 50.0, 80.0])
    query = np.array([[-2.0], [0.0], [2.0]])

    original = photo_ridge_curves(photos, ["A", "B", "C"], curves, query)
    duplicated = photo_ridge_curves(repeated, ["A", "B", "C"], curves, query)

    np.testing.assert_allclose(duplicated, original, atol=1e-12)


def test_within_soil_variation_reduces_prediction_sensitivity():
    ids = ["A", "B", "C"]
    curves = _curves([20.0, 50.0, 80.0])
    stable = _photos({"A": [-1.0], "B": [0.0], "C": [1.0]})
    variable = _photos({"A": [-3.0, 1.0], "B": [-2.0, 2.0], "C": [-1.0, 3.0]})
    query = np.array([[-1.0], [1.0]])

    stable_predictions = photo_ridge_curves(stable, ids, curves, query)
    variable_predictions = photo_ridge_curves(variable, ids, curves, query)

    assert 0 < np.ptp(variable_predictions[:, 0]) < np.ptp(stable_predictions[:, 0])
    np.testing.assert_allclose(variable_predictions[:, 0].mean(), 50.0)


def test_constant_soil_means_keep_the_intercept_unpenalized():
    photos = _photos({"A": [-2.0, 2.0], "B": [0.0], "C": [-1.0, 1.0]})

    predictions = photo_ridge_curves(photos, ["A", "B", "C"], _curves([20.0, 50.0, 80.0]), [[1e6]])

    np.testing.assert_allclose(predictions[0, :10], 50.0)


def test_predictions_are_valid_and_do_not_depend_on_other_queries():
    photos = _photos({"A": [-1.0], "B": [0.0], "C": [1.0]})
    curves = np.array([
        list(np.arange(0.0, 100.0, 10.0)) + [100.0],
        list(np.linspace(20.0, 80.0, 10)) + [100.0],
        list(np.linspace(40.0, 60.0, 10)) + [100.0],
    ])
    alone = photo_ridge_curves(photos, ["A", "B", "C"], curves, [[0.5]])
    together = photo_ridge_curves(photos, ["A", "B", "C"], curves, [[0.5], [-1e6], [1e6]])

    np.testing.assert_allclose(together[0], alone[0])
    assert np.all(np.isfinite(together))
    assert np.all((0 <= together) & (together <= 100))
    assert np.all(np.diff(together, axis=1) >= 0)
    np.testing.assert_array_equal(together[:, -1], 100.0)


def test_evaluation_holds_out_every_photo_and_fits_once_per_soil(monkeypatch):
    photos = _photos({"A": [-3.0, -1.0], "B": [0.0], "C": [1.0, 2.0, 3.0]})
    ids = ["C", "A", "B"]
    curves = _curves([80.0, 20.0, 50.0])
    calls = []

    def inspected_predictor(train_photos, train_ids, train_curves, query_features):
        held_out = ids[len(calls)]
        assert held_out not in train_ids
        assert held_out not in set(train_photos.sample_id)
        assert set(train_photos.sample_id) == set(train_ids)
        np.testing.assert_array_equal(train_curves, curves[[sample != held_out for sample in ids]])
        expected_rows = 1 + photos[photos.sample_id == held_out].camera.nunique()
        assert len(query_features) == expected_rows
        calls.append(held_out)
        return photo_ridge_curves(train_photos, train_ids, train_curves, query_features)

    monkeypatch.setattr("soilgrain.photo_ridge.photo_ridge_curves", inspected_predictor)
    oof, camera_predictions = evaluate_photo_ridge(photos.iloc[::-1], ids, curves)

    assert calls == ids
    assert oof.shape == (3, 11)
    assert len(camera_predictions) == 5
    assert not camera_predictions.duplicated(["sample_id", "camera"]).any()
    assert set(CANONICAL_GRAIN_LABELS).issubset(camera_predictions.columns)


def test_held_out_labels_do_not_affect_their_own_predictions():
    photos = _photos({"A": [-3.0, -1.0], "B": [0.0], "C": [1.0, 2.0, 3.0]})
    curves = _curves([20.0, 50.0, 80.0])
    changed_curves = curves.copy()
    changed_curves[0, :10] = 99.0

    original_oof, original_cameras = evaluate_photo_ridge(photos, ["A", "B", "C"], curves)
    changed_oof, changed_cameras = evaluate_photo_ridge(photos, ["A", "B", "C"], changed_curves)

    np.testing.assert_allclose(original_oof[0], changed_oof[0])
    columns = list(CANONICAL_GRAIN_LABELS)
    np.testing.assert_allclose(
        original_cameras.loc[original_cameras.sample_id == "A", columns],
        changed_cameras.loc[changed_cameras.sample_id == "A", columns],
    )


@pytest.mark.parametrize("problem", [
    "duplicate_ids", "null_id", "missing_soil", "extra_soil", "null_photo_id",
    "duplicate_columns", "no_features", "nan_feature", "bad_curve_shape",
    "nonmonotone_curve", "nan_curve", "wrong_final", "wrong_query_shape", "nan_query",
])
def test_predictor_rejects_invalid_alignment_or_values(problem):
    photos = _photos({"A": [-1.0], "B": [0.0], "C": [1.0]})
    ids = ["A", "B", "C"]
    curves = _curves([20.0, 50.0, 80.0])
    query = np.array([[0.0]])
    if problem == "duplicate_ids":
        ids[1] = "A"
    elif problem == "null_id":
        ids[1] = None
    elif problem == "missing_soil":
        photos = photos.iloc[:2]
    elif problem == "extra_soil":
        photos.loc[2, "sample_id"] = "D"
    elif problem == "null_photo_id":
        photos.loc[1, "sample_id"] = None
    elif problem == "duplicate_columns":
        photos = pd.concat([photos, photos[["feature_0"]]], axis=1)
    elif problem == "no_features":
        photos = photos.drop(columns="feature_0")
    elif problem == "nan_feature":
        photos.loc[0, "feature_0"] = np.nan
    elif problem == "bad_curve_shape":
        curves = curves[:2]
    elif problem == "nonmonotone_curve":
        curves[0, 0] = 30.0
    elif problem == "nan_curve":
        curves[0, 0] = np.nan
    elif problem == "wrong_final":
        curves[0, -1] = 99.0
    elif problem == "wrong_query_shape":
        query = np.zeros((1, 2))
    elif problem == "nan_query":
        query[0, 0] = np.nan

    with pytest.raises(ValueError):
        photo_ridge_curves(photos, ids, curves, query)


@pytest.mark.parametrize("problem", ["single_soil", "missing_camera", "null_camera"])
def test_evaluator_rejects_invalid_grouping(problem):
    photos = _photos({"A": [-1.0], "B": [0.0]})
    ids = ["A", "B"]
    curves = _curves([20.0, 80.0])
    if problem == "single_soil":
        photos, ids, curves = photos.iloc[:1], ids[:1], curves[:1]
    elif problem == "missing_camera":
        photos = photos.drop(columns="camera")
    else:
        photos.loc[0, "camera"] = None

    with pytest.raises(ValueError):
        evaluate_photo_ridge(photos, ids, curves)
