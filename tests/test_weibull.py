from types import SimpleNamespace

import numpy as np
import pytest

from soilgrain.constants import SUPPORT_DIAMETERS
from soilgrain.weibull import decode_weibull, fit_weibull, log_parameter_ridge


def test_recovers_known_weibull_parameters() -> None:
    diameter = np.asarray(SUPPORT_DIAMETERS)
    curve = 100 * (1 - np.exp(-(diameter / 0.8) ** 0.7))
    curve[-1] = 100

    fitted = fit_weibull(curve)

    np.testing.assert_allclose(np.exp(fitted), [0.8, 0.7], rtol=1e-5)
    np.testing.assert_allclose(decode_weibull(fitted), curve, atol=1e-6)


@pytest.mark.parametrize("level", [0., 50., 100.])
def test_fits_flat_extreme_curves_deterministically(level) -> None:
    curve = np.array([level] * 10 + [100.])
    first = fit_weibull(curve)
    second = fit_weibull(curve)

    np.testing.assert_array_equal(first, second)
    decoded = decode_weibull(first)
    assert np.isfinite(decoded).all()
    assert ((decoded >= 0) & (decoded <= 100)).all()
    assert (np.diff(decoded) >= 0).all()
    assert decoded[-1] == 100.


def test_decoding_extreme_log_parameters_is_finite_and_monotone() -> None:
    decoded = decode_weibull(np.array([[-1e300, 1e300], [1e300, -1e300]]))

    assert decoded.shape == (2, 11)
    assert np.isfinite(decoded).all()
    assert ((decoded >= 0) & (decoded <= 100)).all()
    assert (np.diff(decoded, axis=1) >= 0).all()
    np.testing.assert_array_equal(decoded[:, -1], 100.)


def test_failed_optimization_raises_instead_of_substituting_a_curve(monkeypatch) -> None:
    monkeypatch.setattr("soilgrain.weibull.least_squares", lambda *a, **k: SimpleNamespace(
        success=False, x=np.zeros(2), fun=np.zeros(10), message="iteration limit",
    ))

    with pytest.raises(RuntimeError, match="No converged Weibull fit"):
        fit_weibull(np.array([50.] * 10 + [100.]))


def test_parameter_ridge_has_unpenalized_intercept_and_training_only_scaling() -> None:
    features = np.array([[-1.], [1.]])
    targets = np.array([[0., 0.], [2., 0.]])
    prediction = log_parameter_ridge(features, targets, np.array([[1.]]))
    together = log_parameter_ridge(features, targets, np.array([[1.], [1e9]]))
    # Standardized X'X=2 and X'y=2; alpha=10, intercept=1.
    diameter = np.asarray(SUPPORT_DIAMETERS)
    expected = 100 * (1 - np.exp(-diameter / np.exp(7 / 6)))
    expected[-1] = 100

    np.testing.assert_allclose(prediction[0], expected)
    np.testing.assert_allclose(together[0], expected)


@pytest.mark.parametrize("curve", [np.zeros(10), np.array([50.] * 11),
                                    np.array([50., 40.] + [100.] * 9),
                                    np.array([np.nan] + [100.] * 10)])
def test_rejects_invalid_label_curves(curve) -> None:
    with pytest.raises(ValueError, match="curve"):
        fit_weibull(curve)
