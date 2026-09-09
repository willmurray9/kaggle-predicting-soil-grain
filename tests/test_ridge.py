import numpy as np

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
