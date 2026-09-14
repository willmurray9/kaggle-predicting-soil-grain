import numpy as np
import pytest

from soilgrain.ridge import ridge_curves


def test_ridge_curves_matches_hand_calculated_shrinkage() -> None:
    train_features = np.array([[-1.0], [0.0], [1.0]])
    train_curves = np.array([[20.0] * 10 + [100.0], [50.0] * 10 + [100.0], [80.0] * 10 + [100.0]])

    prediction = ridge_curves(train_features, train_curves, np.array([[1.0]]))

    np.testing.assert_allclose(prediction[0, :10], 56.92307692307692)
    assert prediction[0, 10] == 100.0


def test_ridge_scaling_uses_only_training_rows() -> None:
    train_features = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
    train_curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [70.0] * 10 + [100.0]])
    query = np.array([[0.5, 5.0]])

    alone = ridge_curves(train_features, train_curves, query)
    with_extreme_query = ridge_curves(train_features, train_curves, np.vstack([query, [1e9, -1e9]]))

    np.testing.assert_allclose(alone[0], with_extreme_query[0])


def test_ridge_curves_repairs_extrapolated_predictions() -> None:
    train_features = np.array([[-1.0], [0.0], [1.0]])
    train_curves = np.array(
        [
            list(np.arange(0.0, 100.0, 10.0)) + [100.0],
            list(np.linspace(20.0, 80.0, 10)) + [100.0],
            list(np.linspace(40.0, 60.0, 10)) + [100.0],
        ]
    )

    predictions = ridge_curves(train_features, train_curves, np.array([[-1e6], [1e6]]))

    assert predictions.shape == (2, 11)
    assert np.all(np.isfinite(predictions))
    assert np.all((0.0 <= predictions) & (predictions <= 100.0))
    assert np.all(np.diff(predictions, axis=1) >= 0.0)
    np.testing.assert_array_equal(predictions[:, -1], 100.0)


@pytest.mark.parametrize("alpha", [10.0, 100.0, 1000.0])
def test_wide_ridge_matches_primal_solution(alpha: float) -> None:
    rng = np.random.default_rng(42)
    train_features = rng.normal(size=(5, 12))
    train_features[:, -1] = 3.0
    query_features = rng.normal(size=(3, 12))
    train_curves = np.column_stack(
        [np.sort(rng.uniform(0.0, 100.0, size=(5, 10)), axis=1), np.full(5, 100.0)]
    )
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale == 0.0] = 1.0
    train = (train_features - mean) / scale
    query = (query_features - mean) / scale
    target_mean = train_curves[:, :10].mean(axis=0)
    coefficients = np.linalg.solve(
        train.T @ train + alpha * np.eye(train.shape[1]),
        train.T @ (train_curves[:, :10] - target_mean),
    )
    expected = np.maximum.accumulate(np.clip(query @ coefficients + target_mean, 0.0, 100.0), axis=1)

    predictions = ridge_curves(train_features, train_curves, query_features, alpha=alpha)

    np.testing.assert_allclose(predictions[:, :10], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(predictions[:, -1], 100.0)


def test_wide_ridge_predictions_do_not_depend_on_other_queries() -> None:
    train_features = np.array([[0.0, 2.0, 5.0], [1.0, -2.0, 5.0]])
    train_curves = np.array([[20.0] * 10 + [100.0], [80.0] * 10 + [100.0]])
    query = np.array([[0.25, 1.0, 5.0]])

    alone = ridge_curves(train_features, train_curves, query)
    with_extreme_query = ridge_curves(train_features, train_curves, np.vstack([query, [1e9, -1e9, 0.0]]))

    np.testing.assert_allclose(alone[0], with_extreme_query[0])


@pytest.mark.parametrize("shape", [(6, 3), (5, 12)])
def test_full_rank_pca_preserves_ridge_predictions(shape) -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=shape)
    features[:, -1] = 5.0
    curves = np.column_stack([
        np.sort(rng.uniform(0, 100, size=(shape[0], 10)), axis=1),
        np.full(shape[0], 100.0),
    ])
    queries = rng.normal(size=(3, shape[1]))

    original = ridge_curves(features, curves, queries, alpha=100.0)
    compressed = ridge_curves(
        features, curves, queries, alpha=100.0,
        n_components=min(shape[1], shape[0] - 1),
    )

    np.testing.assert_allclose(compressed, original, rtol=1e-12, atol=1e-12)


def test_pca_retains_component_scale_for_ridge_penalty() -> None:
    features = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    curves = np.array([[v] * 10 + [100.0] for v in [20.0, 50.0, 80.0]])

    prediction = ridge_curves(features, curves, np.array([[1.0, 1.0]]), n_components=np.int64(1))

    np.testing.assert_allclose(prediction[0, :10], 61.25)
    assert prediction[0, 10] == 100.0


def test_pca_discards_signal_outside_retained_direction() -> None:
    features = np.array([[-1, -1, -1], [-1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=float)
    curves = np.array([[v] * 10 + [100.0] for v in [20.0, 80.0, 20.0, 80.0]])

    prediction = ridge_curves(features, curves, np.array([[0.0, 0.0, 1.0]]), n_components=1)

    np.testing.assert_allclose(prediction[0, :10], 50.0)


def test_pca_predictions_do_not_depend_on_other_queries() -> None:
    features = np.array([[0.0, 2.0], [1.0, -2.0], [3.0, 1.0]])
    curves = np.array([[v] * 10 + [100.0] for v in [20.0, 50.0, 80.0]])
    query = np.array([[0.25, 1.0]])

    alone = ridge_curves(features, curves, query, n_components=1)
    together = ridge_curves(features, curves, np.vstack([query, [1e9, -1e9]]), n_components=1)

    np.testing.assert_allclose(alone[0], together[0])


@pytest.mark.parametrize("n_components", [0, -1, True, np.bool_(True), 1.5, "1", 3])
def test_pca_rejects_invalid_component_counts(n_components) -> None:
    with pytest.raises(ValueError, match="n_components"):
        ridge_curves(np.zeros((3, 4)), np.zeros((3, 11)), np.zeros((1, 4)), n_components=n_components)


def test_pca_cannot_exceed_input_feature_count() -> None:
    with pytest.raises(ValueError, match="n_components"):
        ridge_curves(np.zeros((4, 2)), np.zeros((4, 11)), np.zeros((1, 2)), n_components=3)
