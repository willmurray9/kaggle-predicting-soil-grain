from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import least_squares

from soilgrain.constants import SUPPORT_DIAMETERS


# Frozen before real-label fitting; bounds are numerical, not image-score tuned.
FIT_RECIPE = {
    "scale_bounds_mm": [1e-5, 1e4],
    "shape_bounds": [0.05, 20.],
    "scale_starts_mm": [0.002, 0.02, 0.2, 2., 20., 200.],
    "shape_starts": [0.25, 1., 4.],
    "method": "trf", "max_nfev": 2000,
    "ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-10,
    "objective": "Sum of log10-support-width weighted squared CDF-percent residuals, first ten supports",
    "convergence": "Choose lowest finite cost among successful starts; stable start-order ties; raise if none",
    "decoding": "Clip predicted log parameters to declared numerical bounds; force final support to 100",
}
LOG_LOWER = np.log([FIT_RECIPE["scale_bounds_mm"][0], FIT_RECIPE["shape_bounds"][0]])
LOG_UPPER = np.log([FIT_RECIPE["scale_bounds_mm"][1], FIT_RECIPE["shape_bounds"][1]])


def decode_weibull(log_parameters: np.ndarray) -> np.ndarray:
    """Decode log(scale, shape) to eleven finite, monotone CDF percentages."""
    parameters = np.asarray(log_parameters, dtype=float)
    if parameters.ndim not in (1, 2) or parameters.shape[-1] != 2 or not np.isfinite(parameters).all():
        raise ValueError("Require finite two-value log parameters")
    parameters = np.clip(parameters, LOG_LOWER, LOG_UPPER)
    log_power = np.exp(parameters[..., 1, None]) * (
        np.log(np.asarray(SUPPORT_DIAMETERS)) - parameters[..., 0, None]
    )
    curves = -100 * np.expm1(-np.exp(np.clip(log_power, -745, np.log(745))))
    curves[..., -1] = 100.
    return curves


def fit_weibull(curve: np.ndarray) -> np.ndarray:
    """Fit one label independently using the predeclared bounded multistart fit."""
    curve = np.asarray(curve, dtype=float)
    if (curve.shape != (11,) or not np.isfinite(curve).all()
            or ((curve < 0) | (curve > 100)).any() or (np.diff(curve) < -1e-9).any()
            or not np.isclose(curve[-1], 100., rtol=0, atol=1e-9)):
        raise ValueError("Require one finite cumulative curve of eleven percentages ending at 100")
    weights = np.sqrt(np.diff(np.log10(SUPPORT_DIAMETERS)))

    def residual(parameters):
        return weights * (decode_weibull(parameters)[:10] - curve[:10])

    best, best_cost = None, np.inf
    for scale, shape in product(FIT_RECIPE["scale_starts_mm"], FIT_RECIPE["shape_starts"]):
        fit = least_squares(
            residual, np.log([scale, shape]), bounds=(LOG_LOWER, LOG_UPPER),
            **{key: FIT_RECIPE[key] for key in ("method", "max_nfev", "ftol", "xtol", "gtol")},
        )
        if fit.success and np.isfinite(fit.x).all() and np.isfinite(fit.fun).all():
            cost = float(np.square(fit.fun).sum())
            if cost < best_cost:
                best, best_cost = fit.x.copy(), cost
    if best is None:
        raise RuntimeError("No converged Weibull fit among the declared starting points")
    return best


def log_parameter_ridge(
    train_features: np.ndarray, train_parameters: np.ndarray, query_features: np.ndarray,
) -> np.ndarray:
    """Training-only standardization and alpha=10 ridge with an unpenalized intercept."""
    features, parameters, queries = [np.asarray(a, dtype=float) for a in
                                     (train_features, train_parameters, query_features)]
    if (features.ndim != 2 or queries.ndim != 2 or min(features.shape) == 0
            or queries.shape[1] != features.shape[1] or parameters.shape != (len(features), 2)
            or not all(np.isfinite(a).all() for a in (features, parameters, queries))):
        raise ValueError("Require finite matching features and two log parameters per training soil")
    mean, std = features.mean(axis=0), features.std(axis=0)
    std[std == 0] = 1.
    scaled_train, scaled_query = (features - mean) / std, (queries - mean) / std
    intercept = parameters.mean(axis=0)
    coefficients = np.linalg.solve(
        scaled_train.T @ scaled_train + 10. * np.eye(features.shape[1]),
        scaled_train.T @ (parameters - intercept),
    )
    return decode_weibull(scaled_query @ coefficients + intercept)
