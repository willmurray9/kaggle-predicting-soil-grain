import numpy as np
import pytest

from soilgrain.distribution_models import emd_median_regression, sqrt_mass_ridge


def test_sqrt_mass_decoding_uses_squared_roots_and_normalizes():
    features = np.array([[-2.], [2.]])
    curves = np.array([[100.] * 11, [0.] * 10 + [100.]])

    prediction = sqrt_mass_ridge(features, curves, np.array([[4.]]))

    # Scaled query=2; ridge roots=(1/3, 2/3); normalized squares=(1/5, 4/5).
    np.testing.assert_allclose(prediction, [[20.] * 10 + [100.]])


def test_sqrt_mass_zero_roots_fall_back_to_training_mean_masses():
    masses = np.vstack([np.eye(11), np.eye(11)[0], np.full(11, 1 / 11)])
    curves = np.cumsum(masses, axis=1) * 100
    curves[:, -1] = 100.
    features = np.r_[np.zeros(12), 1.][:, None]

    prediction = sqrt_mass_ridge(features, curves, np.array([[-100.]]))

    # Every root extrapolates below zero. Mean masses are 23/143, then ten 12/143 bins.
    np.testing.assert_allclose(prediction[0], (23 + 12 * np.arange(11)) / 143 * 100)


def test_median_regression_uses_training_median_and_unpenalized_intercept():
    features = np.full((3, 2), 5.)
    curves = np.array([[v] * 10 + [100.] for v in [10., 20., 90.]])

    predictions = emd_median_regression(features, curves, np.array([[5., 5.], [100., -100.]]))

    np.testing.assert_allclose(predictions, [[20.] * 10 + [100.]] * 2)


def test_median_regression_recovers_exact_linear_conditional_medians():
    features = np.array([[-2.], [2.]])
    curves = np.array([[v] * 10 + [100.] for v in [20., 80.]])

    predictions = emd_median_regression(features, curves, np.array([[-1.], [1.]]))

    np.testing.assert_allclose(predictions, [[35.] * 10 + [100.], [65.] * 10 + [100.]])


@pytest.mark.parametrize("predictor", [sqrt_mass_ridge, emd_median_regression])
def test_scaling_uses_training_rows_and_predictions_ignore_other_queries(predictor):
    features = np.array([[0., 5.], [1., 5.], [4., 5.]])
    curves = np.array([[v] * 10 + [100.] for v in [10., 40., 90.]])
    query = np.array([[0.5, 5.]])

    alone = predictor(features, curves, query)
    together = predictor(features, curves, np.vstack([query, [1e9, -1e9]]))
    rescaled = predictor(features * [100., 2.] + 1000., curves, query * [100., 2.] + 1000.)

    np.testing.assert_allclose(together[:1], alone)
    np.testing.assert_allclose(rescaled, alone)
    assert np.isfinite(together).all()
    assert ((together >= 0) & (together <= 100)).all()
    assert (np.diff(together, axis=1) >= 0).all()
    np.testing.assert_array_equal(together[:, -1], 100.)


@pytest.mark.parametrize("predictor", [sqrt_mass_ridge, emd_median_regression])
def test_constant_targets_and_empty_queries(predictor):
    curve = np.arange(0., 101., 10.)
    prediction = predictor(np.array([[0.], [2.]]), np.tile(curve, (2, 1)), np.array([[-1e9], [1e9]]))

    np.testing.assert_allclose(prediction, np.tile(curve, (2, 1)))
    assert predictor(np.zeros((1, 1)), curve[None], np.empty((0, 1))).shape == (0, 11)


@pytest.mark.parametrize("predictor", [sqrt_mass_ridge, emd_median_regression])
@pytest.mark.parametrize("problem", ["dimensions", "empty", "feature_count", "curve_count", "nan", "descending", "negative", "final"])
def test_invalid_inputs_fail_explicitly(predictor, problem):
    features, query = np.zeros((2, 1)), np.zeros((1, 1))
    curves = np.tile(np.arange(0., 101., 10.), (2, 1))
    if problem == "dimensions":
        features = features.ravel()
    elif problem == "empty":
        features, curves = features[:0], curves[:0]
    elif problem == "feature_count":
        query = np.zeros((1, 2))
    elif problem == "curve_count":
        curves = curves[:1]
    elif problem == "nan":
        query[0, 0] = np.nan
    elif problem == "descending":
        curves[0, 3] = 5.
    elif problem == "negative":
        curves[0, 0] = -1.
    elif problem == "final":
        curves[:, -1] = 99.

    with pytest.raises(ValueError):
        predictor(features, curves, query)
