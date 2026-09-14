import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

import soilgrain.svr as svr


@pytest.mark.parametrize("C,slope", [(0.1, 0.2), (1.0, 2.0), (10.0, 9.0)])
def test_svr_matches_two_soil_epsilon_shrinkage(C, slope):
    features = np.array([[-2.0, 7.0], [2.0, 7.0]])
    curves = np.array([[40.0] * 10 + [100.0], [60.0] * 10 + [100.0]])
    queries = np.array([[-2.0, 7.0], [0.0, 7.0], [2.0, 7.0]])

    predictions = svr.svr_curves(features, curves, queries, C=C)

    # Standardized x is +/-1: minimize w^2/2 + 2*C*max(9-w, 0).
    expected = np.repeat(np.array([[50.0 - slope], [50.0], [50.0 + slope]]), 10, axis=1)
    np.testing.assert_allclose(predictions[:, :10], expected, atol=1e-6)
    np.testing.assert_array_equal(predictions[:, 10], 100.0)


def test_svr_intercept_is_not_penalized_and_query_rows_do_not_fit_scaling():
    features = np.array([[0.0, 5.0], [1.0, 5.0], [4.0, 5.0]])
    curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [70.0] * 10 + [100.0]])
    query = np.array([[0.5, 5.0]])
    original = svr.svr_curves(features, curves, query)
    batched = svr.svr_curves(features, curves, np.vstack([query, [1e9, -1e9]]))
    shifted_curves = curves.copy()
    shifted_curves[:, :10] += 10.0
    shifted = svr.svr_curves(features, shifted_curves, query)

    np.testing.assert_allclose(original[0], batched[0], atol=1e-10)
    np.testing.assert_allclose(shifted[:, :10], original[:, :10] + 10.0, atol=1e-6)


def test_svr_preserves_constant_targets_and_one_soil_training():
    curve = np.arange(0.0, 101.0, 10.0)
    queries = np.array([[0.5], [1e9]])
    for features in (np.array([[0.0], [1.0], [4.0]]), np.array([[2.0]])):
        predictions = svr.svr_curves(features, np.tile(curve, (len(features), 1)), queries)
        np.testing.assert_allclose(predictions, np.tile(curve, (2, 1)), atol=1e-12)


def test_svr_constant_features_fit_robust_intercept():
    features = np.full((3, 2), 5.0)
    curves = np.array([[10.0] * 10 + [100.0], [40.0] * 10 + [100.0], [90.0] * 10 + [100.0]])

    predictions = svr.svr_curves(features, curves, np.array([[5.0, 5.0], [100.0, -100.0]]))

    # The optimum intercept lies in [39,41], the middle target's epsilon tube.
    assert np.all((39.0 <= predictions[:, :10]) & (predictions[:, :10] <= 41.0))
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_svr_keeps_existing_curve_repair():
    features = np.array([[-1.0], [1.0]])
    curves = np.tile([-10, 20, 10, 30, 110, 90, 80, 70, 60, 50, 100], (2, 1))

    predictions = svr.svr_curves(features, curves, np.array([[0.0]]))

    np.testing.assert_allclose(predictions, [[0, 20, 20, 30, 100, 100, 100, 100, 100, 100, 100]])


@pytest.mark.parametrize("C", [0.0, -1.0, np.nan, np.inf])
def test_svr_rejects_invalid_penalty(C):
    with pytest.raises(ValueError, match="C"):
        svr.svr_curves(np.zeros((2, 1)), np.zeros((2, 11)), np.zeros((1, 1)), C=C)


@pytest.mark.parametrize("features,curves,query", [
    (np.zeros(2), np.zeros((2, 11)), np.zeros((1, 1))),
    (np.zeros((0, 1)), np.zeros((0, 11)), np.zeros((1, 1))),
    (np.zeros((2, 1)), np.zeros((2, 10)), np.zeros((1, 1))),
    (np.zeros((2, 1)), np.zeros((2, 11)), np.zeros((1, 2))),
    (np.zeros((2, 1)), np.zeros((2, 11)), np.full((1, 1), np.nan)),
])
def test_svr_rejects_misaligned_or_nonfinite_inputs(features, curves, query):
    with pytest.raises(ValueError):
        svr.svr_curves(features, curves, query)


@pytest.mark.parametrize("failure", ["warning", "status", "nonfinite"])
def test_svr_rejects_solver_failures(monkeypatch, failure):
    class FailedSolver:
        def __init__(self, **kwargs):
            self.fit_status_ = int(failure == "status")

        def fit(self, x, y):
            if failure == "warning":
                warnings.warn("iteration limit", ConvergenceWarning)
            return self

        def predict(self, query):
            return np.full(len(query), np.nan if failure == "nonfinite" else 50.0)

    monkeypatch.setattr(svr, "SVR", FailedSolver)
    with pytest.raises((RuntimeError, ConvergenceWarning)):
        svr.svr_curves(np.zeros((2, 1)), np.zeros((2, 11)), np.zeros((1, 1)))
