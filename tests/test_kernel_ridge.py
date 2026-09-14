import numpy as np
import pytest

from soilgrain.kernel_ridge import kernel_ridge_curves


def test_kernel_ridge_matches_two_soil_shrinkage() -> None:
    train_features = np.array([[-1.0, 9.0], [1.0, 9.0]])
    train_curves = np.array([[20.0] * 10 + [100.0], [80.0] * 10 + [100.0]])

    prediction = kernel_ridge_curves(train_features, train_curves, train_features[1:])

    # The centered two-soil kernel has eigenvalue 1 - exp(-4 / 2).
    eigenvalue = 1.0 - np.exp(-2.0)
    expected = 50.0 + 30.0 * eigenvalue / (eigenvalue + 1.0)
    np.testing.assert_allclose(prediction[0, :10], expected)
    assert prediction[0, 10] == 100.0


@pytest.mark.parametrize("gamma", [None, 0.07])
def test_kernel_ridge_matches_unpenalized_intercept_block_system(gamma) -> None:
    train_features = np.array([[0.0, 0.0], [2.0, 1.0], [3.0, -2.0], [7.0, 4.0]])
    train_curves = np.array(
        [
            list(np.linspace(12, 45, 10)) + [100.0],
            list(np.linspace(20, 65, 10)) + [100.0],
            list(np.linspace(35, 78, 10)) + [100.0],
            list(np.linspace(50, 90, 10)) + [100.0],
        ]
    )
    queries = np.array([[1.0, 3.0], [5.0, -1.0]])
    variance = train_features.var(axis=0)
    alpha = 0.7
    bandwidth = 0.5 if gamma is None else gamma

    # Solve the original kernel and explicit intercept jointly, without centering.
    kernel = np.array(
        [
            [np.exp(-bandwidth * np.sum((left - right) ** 2 / variance)) for right in train_features]
            for left in train_features
        ]
    )
    system = np.block(
        [
            [kernel + alpha * np.eye(4), np.ones((4, 1))],
            [np.ones((1, 4)), np.zeros((1, 1))],
        ]
    )
    solution = np.linalg.solve(system, np.vstack([train_curves[:, :10], np.zeros(10)]))
    query_kernel = np.array(
        [
            [np.exp(-bandwidth * np.sum((query - row) ** 2 / variance)) for row in train_features]
            for query in queries
        ]
    )
    expected = query_kernel @ solution[:4] + solution[4]
    assert np.all((0.0 < expected) & (expected < 100.0))
    assert np.all(np.diff(expected, axis=1) >= 0.0)

    predictions = kernel_ridge_curves(train_features, train_curves, queries, alpha=alpha, gamma=gamma)

    np.testing.assert_allclose(predictions[:, :10], expected, atol=1e-12)


def test_default_gamma_matches_explicit_inverse_feature_count() -> None:
    features = np.array([[0.0, 3.0], [2.0, 1.0], [5.0, 7.0]])
    curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [70.0] * 10 + [100.0]])
    queries = np.array([[1.0, 5.0]])

    implicit = kernel_ridge_curves(features, curves, queries)
    explicit = kernel_ridge_curves(features, curves, queries, gamma=0.5)

    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize("gamma", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_gamma_must_be_positive_and_finite(gamma) -> None:
    features = np.array([[0.0], [1.0]])
    curves = np.tile([25.0] * 10 + [100.0], (2, 1))

    with pytest.raises(ValueError, match="gamma"):
        kernel_ridge_curves(features, curves, features[:1], gamma=gamma)


def test_kernel_ridge_scaling_and_centering_use_only_training_rows() -> None:
    train_features = np.array([[0.0, 5.0], [1.0, 5.0], [4.0, 5.0]])
    train_curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [70.0] * 10 + [100.0]])
    query = np.array([[0.5, 5.0]])

    alone = kernel_ridge_curves(train_features, train_curves, query)
    with_extreme_query = kernel_ridge_curves(
        train_features, train_curves, np.vstack([query, [1e9, -1e9]])
    )

    np.testing.assert_allclose(alone[0], with_extreme_query[0], atol=1e-12)


def test_kernel_ridge_constant_features_predict_target_mean() -> None:
    train_features = np.full((3, 2), 5.0)
    train_curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [70.0] * 10 + [100.0]])

    predictions = kernel_ridge_curves(train_features, train_curves, np.array([[5.0, 5.0], [100.0, -100.0]]))

    np.testing.assert_allclose(predictions[:, :10], 40.0)


def test_kernel_ridge_constant_targets_preserve_unpenalized_mean() -> None:
    train_features = np.array([[0.0], [1.0], [4.0]])
    curve = np.arange(0.0, 101.0, 10.0)
    train_curves = np.tile(curve, (3, 1))

    predictions = kernel_ridge_curves(train_features, train_curves, np.array([[0.5], [1e9]]))

    np.testing.assert_allclose(predictions, np.tile(curve, (2, 1)), atol=1e-12)


def test_kernel_ridge_repairs_out_of_range_and_decreasing_predictions() -> None:
    train_features = np.array([[-1.0], [1.0]])
    train_curves = np.tile([-10.0, 20.0, 10.0, 30.0, 110.0, 90.0, 80.0, 70.0, 60.0, 50.0, 100.0], (2, 1))

    predictions = kernel_ridge_curves(train_features, train_curves, np.array([[0.0]]))

    np.testing.assert_allclose(predictions, [[0.0, 20.0, 20.0, 30.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]])
