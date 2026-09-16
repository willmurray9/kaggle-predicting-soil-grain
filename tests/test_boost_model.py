import numpy as np
import pandas as pd
import pytest

import soilgrain.boost_model as boost_model
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples


def test_boost_matches_analytic_fixed_stage_shrinkage():
    features = np.repeat([[-1.0], [1.0]], 4, axis=0)
    curves = np.column_stack([np.repeat([[20.0], [80.0]], 4, axis=0).repeat(10, axis=1), np.full(8, 100.0)])

    predictions = boost_model.boost_curves(features, curves, np.array([[-1.0], [1.0]]))

    expected = 50.0 + np.array([-30.0, 30.0]) * (1.0 - 0.97 ** 128)
    np.testing.assert_allclose(predictions[:, :10], np.repeat(expected[:, None], 10, axis=1))
    np.testing.assert_array_equal(predictions[:, 10], 100.0)


def test_constant_features_predict_training_mean():
    curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [90.0] * 10 + [100.0]])

    predictions = boost_model.boost_curves(np.ones((3, 2)), curves, np.array([[1.0, 1.0], [1e9, -1e9]]))

    np.testing.assert_allclose(predictions, np.tile(curves.mean(axis=0), (2, 1)))


def test_boost_is_repeatable_query_independent_and_valid():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(12, 3))
    curves = np.column_stack([np.sort(rng.uniform(0, 100, size=(12, 10)), axis=1), np.full(12, 100.0)])
    query = np.array([[0.5, -0.5, 1.0]])

    prediction = boost_model.boost_curves(features, curves, query)
    repeated = boost_model.boost_curves(features, curves, query)
    batched = boost_model.boost_curves(features, curves, np.vstack([query, [1e9, -1e9, 0.0]]))

    np.testing.assert_array_equal(prediction, repeated)
    np.testing.assert_array_equal(prediction[0], batched[0])
    assert np.all(np.isfinite(batched))
    assert np.all(np.diff(batched, axis=1) >= 0.0)
    assert np.all((0.0 <= batched) & (batched <= 100.0))
    np.testing.assert_array_equal(batched[:, 10], 100.0)


def test_boost_fits_ten_targets_with_fixed_depth_and_leaf_limits(monkeypatch):
    fitted = []
    real_regressor = boost_model.GradientBoostingRegressor

    def capture_regressor(**parameters):
        model = real_regressor(**parameters)
        fitted.append(model)
        return model

    monkeypatch.setattr(boost_model, "GradientBoostingRegressor", capture_regressor)
    rng = np.random.default_rng(1)
    features = rng.normal(size=(24, 23))
    curves = np.column_stack([np.sort(rng.uniform(0, 100, size=(24, 10)), axis=1), np.full(24, 100.0)])

    boost_model.boost_curves(features, curves, features[:1])

    assert len(fitted) == 10
    for column, model in enumerate(fitted):
        assert model.n_estimators_ == 128
        assert model.n_features_in_ == 23
        np.testing.assert_allclose(model.init_.constant_, curves[:, column].mean())
        for tree in model.estimators_[:, 0]:
            assert tree.get_depth() <= 2
            leaves = tree.tree_.children_left == -1
            assert np.all(tree.tree_.n_node_samples[leaves] >= 4)


def test_boost_evaluation_excludes_entire_held_out_soil():
    photos = pd.DataFrame({
        "sample_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "camera": ["Phone 1", "Phone 2"] * 4,
        "feature_0": [0, 100, 1, 101, 2, 102, 3, 103],
    })
    curves = np.column_stack([np.repeat([[0], [30], [60], [90]], 10, axis=1), [100] * 4])

    oof, views = evaluate_samples(photos, ["A", "B", "C", "D"], curves, boost_model.boost_curves)

    # Three training soils cannot split under the declared four-soil minimum leaf.
    np.testing.assert_allclose(oof[:, 0], [60, 50, 40, 30])
    for sample_id, expected in zip(["A", "B", "C", "D"], [60, 50, 40, 30], strict=True):
        rows = views[views["sample_id"] == sample_id]
        assert len(rows) == 2
        np.testing.assert_allclose(rows[list(CANONICAL_GRAIN_LABELS)[:10]], expected)


def test_boost_repairs_curves_and_accepts_empty_queries():
    curves = np.tile([-10, 20, 10, 30, 110, 90, 80, 70, 60, 50, 100], (3, 1))

    prediction = boost_model.boost_curves(np.ones((3, 1)), curves, np.array([[0.0]]))
    empty = boost_model.boost_curves(np.ones((3, 1)), curves, np.empty((0, 1)))

    np.testing.assert_allclose(prediction, [[0, 20, 20, 30, 100, 100, 100, 100, 100, 100, 100]])
    assert empty.shape == (0, 11)


@pytest.mark.parametrize("features,curves,query", [
    (np.zeros(2), np.zeros((2, 11)), np.zeros((1, 1))),
    (np.zeros((0, 1)), np.zeros((0, 11)), np.zeros((1, 1))),
    (np.zeros((2, 0)), np.zeros((2, 11)), np.zeros((1, 0))),
    (np.zeros((2, 1)), np.zeros((2, 10)), np.zeros((1, 1))),
    (np.zeros((2, 1)), np.zeros((2, 11)), np.zeros((1, 2))),
    (np.full((2, 1), np.inf), np.zeros((2, 11)), np.zeros((1, 1))),
    (np.zeros((2, 1)), np.full((2, 11), np.nan), np.zeros((1, 1))),
    (np.zeros((2, 1)), np.zeros((2, 11)), np.full((1, 1), np.nan)),
])
def test_boost_rejects_misaligned_or_nonfinite_inputs(features, curves, query):
    with pytest.raises(ValueError):
        boost_model.boost_curves(features, curves, query)
