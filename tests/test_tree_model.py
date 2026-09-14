import numpy as np
import pandas as pd
import pytest

import soilgrain.tree_model as tree_model
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples


def test_tree_predictions_match_known_leaf_means():
    features = np.repeat([[-1.0], [1.0]], 3, axis=0)
    curves = np.column_stack([
        np.repeat(np.array([[10], [20], [30], [70], [80], [90]]), 10, axis=1),
        np.full(6, 100.0),
    ])

    predictions = tree_model.extra_trees_curves(features, curves, np.array([[-1.0], [1.0]]))

    np.testing.assert_allclose(predictions[:, :10], np.repeat([[20.0], [80.0]], 10, axis=1))
    np.testing.assert_array_equal(predictions[:, 10], 100.0)


def test_constant_features_predict_training_mean():
    features = np.ones((3, 2))
    curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [90.0] * 10 + [100.0]])

    predictions = tree_model.extra_trees_curves(features, curves, np.array([[1.0, 1.0], [1e9, -1e9]]))

    np.testing.assert_allclose(predictions, np.tile(curves.mean(axis=0), (2, 1)))


def test_tree_predictions_are_repeatable_independent_of_other_queries_and_valid():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(12, 3))
    curves = np.column_stack([np.sort(rng.uniform(0, 100, size=(12, 10)), axis=1), np.full(12, 100.0)])
    query = np.array([[0.5, -0.5, 1.0]])

    prediction = tree_model.extra_trees_curves(features, curves, query)
    repeated = tree_model.extra_trees_curves(features, curves, query)
    batched = tree_model.extra_trees_curves(features, curves, np.vstack([query, [1e9, -1e9, 0.0]]))

    np.testing.assert_array_equal(prediction, repeated)
    np.testing.assert_array_equal(prediction[0], batched[0])
    assert np.all(np.diff(batched, axis=1) >= 0.0)
    assert np.all((0.0 <= batched) & (batched <= 100.0))
    np.testing.assert_array_equal(batched[:, 10], 100.0)


def test_fitted_forest_shares_outputs_and_limits_depth_and_leaf_size(monkeypatch):
    fitted = []
    real_forest = tree_model.ExtraTreesRegressor

    def capture_forest(**parameters):
        model = real_forest(**parameters)
        fitted.append(model)
        return model

    monkeypatch.setattr(tree_model, "ExtraTreesRegressor", capture_forest)
    rng = np.random.default_rng(1)
    features = rng.normal(size=(24, 17))
    curves = np.column_stack([np.sort(rng.uniform(0, 100, size=(24, 10)), axis=1), np.full(24, 100.0)])

    tree_model.extra_trees_curves(features, curves, features[:1])

    assert len(fitted) == 1
    assert fitted[0].n_outputs_ == 10
    assert len(fitted[0].estimators_) == 256
    for estimator in fitted[0].estimators_:
        assert estimator.get_depth() <= 3
        leaves = estimator.tree_.children_left == -1
        assert np.all(estimator.tree_.n_node_samples[leaves] >= 3)


def test_tree_evaluation_excludes_held_out_soil_from_pooled_and_camera_predictions():
    photos = pd.DataFrame({
        "sample_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
        "camera": ["Phone 1", "Phone 2"] * 4,
        "feature_0": [0, 100, 1, 101, 2, 102, 3, 103],
    })
    curves = np.column_stack([np.repeat([[0], [30], [60], [90]], 10, axis=1), [100] * 4])

    oof, camera_predictions = evaluate_samples(photos, ["A", "B", "C", "D"], curves, tree_model.extra_trees_curves)

    # Each fold has three training soils: the minimum leaf size forces their mean.
    np.testing.assert_allclose(oof[:, 0], [60, 50, 40, 30])
    for sample_id, expected in zip(["A", "B", "C", "D"], [60, 50, 40, 30], strict=True):
        rows = camera_predictions[camera_predictions["sample_id"] == sample_id]
        assert len(rows) == 2
        np.testing.assert_allclose(rows[list(CANONICAL_GRAIN_LABELS)[:10]], expected)


def test_tree_keeps_existing_curve_repair():
    curves = np.tile([-10, 20, 10, 30, 110, 90, 80, 70, 60, 50, 100], (3, 1))

    predictions = tree_model.extra_trees_curves(np.ones((3, 1)), curves, np.array([[0.0]]))

    np.testing.assert_allclose(predictions, [[0, 20, 20, 30, 100, 100, 100, 100, 100, 100, 100]])


def test_tree_accepts_empty_queries():
    predictions = tree_model.extra_trees_curves(np.ones((3, 2)), np.zeros((3, 11)), np.empty((0, 2)))
    assert predictions.shape == (0, 11)


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
def test_tree_rejects_misaligned_or_nonfinite_inputs(features, curves, query):
    with pytest.raises(ValueError):
        tree_model.extra_trees_curves(features, curves, query)
