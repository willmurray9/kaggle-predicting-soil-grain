import numpy as np
import pytest

from soilgrain.constants import SUPPORT_DIAMETERS
from soilgrain.metrics import emd_score
from soilgrain.transport_ridge import (
    curves_to_log_quantiles,
    log_quantiles_to_curves,
    transport_ridge_curves,
)


def _point_mass(index):
    curve = np.zeros(11)
    curve[index:] = 100.0
    return curve


def test_discrete_inverse_uses_first_support_at_each_midpoint_percentile():
    curve = np.array([[0.05, 0.05, 25.0, 25.0, 75.0, 75.0, 100.0, 100.0, 100.0, 100.0, 100.0]])

    quantiles = curves_to_log_quantiles(curve)

    assert quantiles.shape == (1, 1000)
    np.testing.assert_array_equal(quantiles[0, :1], np.log10(0.002))
    np.testing.assert_array_equal(quantiles[0, 1:250], np.log10(0.02))
    np.testing.assert_array_equal(quantiles[0, 250:750], np.log10(0.2))
    np.testing.assert_array_equal(quantiles[0, 750:], np.log10(2.0))


def test_discrete_round_trip_respects_fixed_quantization_bound():
    rng = np.random.default_rng(28)
    curves = np.column_stack([np.sort(rng.uniform(0, 100, (20, 10)), axis=1), np.full(20, 100.0)])
    curves = np.vstack([curves, [0, 0, 10.05, 10.05, 33.333, 50.15, 50.15, 88.888, 99.95, 100, 100]])

    reconstructed = log_quantiles_to_curves(curves_to_log_quantiles(curves))

    assert np.max(np.abs(reconstructed - curves)) <= 0.05 + 1e-12
    for actual, expected in zip(reconstructed, curves, strict=True):
        assert emd_score(expected, actual) <= 0.25 + 1e-12
    np.testing.assert_array_equal(reconstructed[:, -1], 100.0)


def test_equal_log_transport_moves_mass_to_geometric_midpoint():
    curves = np.vstack([_point_mass(0), _point_mass(4)])

    predictions = transport_ridge_curves(np.zeros((2, 1)), curves, [[0.0]])

    np.testing.assert_array_equal(predictions[0], _point_mass(2))
    assert not np.array_equal(predictions[0], curves.mean(axis=0))


def test_nonmonotone_prediction_uses_least_squares_isotonic_projection():
    central = _point_mass(2)
    spread = (_point_mass(0) + _point_mass(4)) / 2
    curves = np.vstack([central, spread])

    prediction = transport_ridge_curves([[-1.0], [1.0]], curves, [[-12.0]])

    # Alpha 10 gives raw quantiles at log10(.02)+.5 for the first half,
    # then log10(.02)-.5. Isotonic projection pools both halves at log10(.02).
    # Sorting retains two masses; cumulative maximum retains the larger value.
    np.testing.assert_array_equal(prediction[0], central)


@pytest.mark.parametrize("support_index", range(11))
def test_constant_point_mass_survives_floating_point_fit_and_projection(support_index):
    curves = np.tile(_point_mass(support_index), (23, 1))
    features = np.arange(23, dtype=float)[:, None]

    prediction = transport_ridge_curves(features, curves, [[11.0]])

    np.testing.assert_array_equal(prediction[0], curves[0])


def test_inverse_tolerance_only_absorbs_numerical_support_equality():
    exact = np.log10(0.063)
    quantiles = np.vstack([np.full(1000, exact + 1e-14), np.full(1000, exact + 1e-8)])

    curves = log_quantiles_to_curves(quantiles)

    np.testing.assert_array_equal(curves[0], _point_mass(3))
    np.testing.assert_array_equal(curves[1], _point_mass(4))


def test_training_scaling_and_alpha_match_hand_calculated_shrinkage():
    curves = np.vstack([_point_mass(0), _point_mass(2), _point_mass(4)])
    query = [[13 / 3 * 0.99]]

    prediction = transport_ridge_curves([[-1.0], [0.0], [1.0]], curves, query)

    # Standardized X'X=3. Raw log-diameter slope is 3/(3+10), so the
    # query gives log10(.02)+.99, immediately below the .2 mm support.
    np.testing.assert_array_equal(prediction[0], _point_mass(4))


def test_query_cohort_and_affine_input_scaling_do_not_change_predictions():
    features = np.array([[-1.0, 5.0], [0.0, 5.0], [1.0, 5.0]])
    curves = np.vstack([_point_mass(0), _point_mass(2), _point_mass(4)])
    query = np.array([[13 / 3 * 0.99, 5.0]])

    alone = transport_ridge_curves(features, curves, query)
    mixed = transport_ridge_curves(features, curves, np.vstack([query, [1e6, -1e6]]))
    transformed = transport_ridge_curves(features * 7 + 9, curves, query * 7 + 9)

    np.testing.assert_array_equal(alone[0], mixed[0])
    np.testing.assert_array_equal(transformed, alone)


def test_extreme_predictions_are_bounded_valid_curves():
    curves = np.vstack([_point_mass(0), _point_mass(4)])

    predictions = transport_ridge_curves([[-1.0], [1.0]], curves, [[-1e6], [1e6]])

    np.testing.assert_array_equal(predictions[0], _point_mass(0))
    np.testing.assert_array_equal(predictions[1], _point_mass(10))
    assert np.all(np.isfinite(predictions))
    assert np.all((predictions >= 0) & (predictions <= 100))
    assert np.all(np.diff(predictions, axis=1) >= 0)


def test_empty_query_preserves_output_shape():
    predictions = transport_ridge_curves([[0.0]], [_point_mass(3)], np.empty((0, 1)))

    assert predictions.shape == (0, 11)


@pytest.mark.parametrize("problem", ["wrong_shape", "nan", "negative", "above_100", "decreasing", "incomplete"])
def test_curve_conversion_rejects_invalid_cdfs(problem):
    curves = np.array([[20.0] * 10 + [100.0]])
    if problem == "wrong_shape":
        curves = curves[:, :10]
    elif problem == "nan":
        curves[0, 0] = np.nan
    elif problem == "negative":
        curves[0, 0] = -1.0
    elif problem == "above_100":
        curves[0, -1] = 101.0
    elif problem == "decreasing":
        curves[0, 0] = 30.0
    else:
        curves[0, -1] = 99.0

    with pytest.raises(ValueError):
        curves_to_log_quantiles(curves)


@pytest.mark.parametrize("problem", ["wrong_shape", "nan", "decreasing", "outside_support"])
def test_inverse_conversion_rejects_invalid_quantiles(problem):
    quantiles = np.zeros((1, 1000))
    if problem == "wrong_shape":
        quantiles = quantiles[:, :-1]
    elif problem == "nan":
        quantiles[0, 0] = np.nan
    elif problem == "decreasing":
        quantiles[0, 0] = 1.0
    else:
        quantiles[:] = np.log10(SUPPORT_DIAMETERS[-1]) + 1

    with pytest.raises(ValueError):
        log_quantiles_to_curves(quantiles)


@pytest.mark.parametrize("problem", ["empty_training", "empty_features", "curve_count", "query_dimensions", "nan_training", "nan_query"])
def test_predictor_rejects_invalid_training_or_query_inputs(problem):
    features = np.array([[-1.0], [1.0]])
    curves = np.vstack([_point_mass(0), _point_mass(4)])
    query = np.array([[0.0]])
    if problem == "empty_training":
        features, curves = features[:0], curves[:0]
    elif problem == "empty_features":
        features, query = features[:, :0], query[:, :0]
    elif problem == "curve_count":
        curves = curves[:1]
    elif problem == "query_dimensions":
        query = np.zeros((1, 2))
    elif problem == "nan_training":
        features[0, 0] = np.nan
    else:
        query[0, 0] = np.nan

    with pytest.raises(ValueError):
        transport_ridge_curves(features, curves, query)
