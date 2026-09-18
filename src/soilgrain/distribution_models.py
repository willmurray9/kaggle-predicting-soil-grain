import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor, Ridge


def _scaled_inputs(train_features, train_curves, query_features):
    features, curves, queries = [np.asarray(a, dtype=float) for a in
                                (train_features, train_curves, query_features)]
    if (features.ndim != 2 or curves.ndim != 2 or queries.ndim != 2
            or min(features.shape) == 0 or curves.shape != (len(features), 11)
            or queries.shape[1] != features.shape[1]):
        raise ValueError("Require nonempty matching features and eleven-value training curves")
    if not all(np.isfinite(a).all() for a in (features, curves, queries)):
        raise ValueError("Features and curves must be finite")
    if (((curves < 0) | (curves > 100)).any() or (np.diff(curves, axis=1) < 0).any()
            or not np.all(curves[:, -1] == 100)):
        raise ValueError("Training curves must be cumulative percentages ending at 100")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1.
    return (features - mean) / scale, curves, (queries - mean) / scale


def sqrt_mass_ridge(
    train_features: np.ndarray, train_curves: np.ndarray, query_features: np.ndarray,
) -> np.ndarray:
    """Fixed alpha=10 ridge on square-root mass; decode to the mass simplex."""
    features, curves, queries = _scaled_inputs(train_features, train_curves, query_features)
    if len(queries) == 0:
        return np.empty((0, 11))
    masses = np.diff(np.column_stack([np.zeros(len(curves)), curves]), axis=1) / 100.
    model = Ridge(alpha=10., fit_intercept=True).fit(features, np.sqrt(masses))
    with np.errstate(over="raise", invalid="raise"):
        predicted = np.square(np.maximum(model.predict(queries), 0.))
    if not np.isfinite(predicted).all():
        raise RuntimeError("Square-root mass predictions must be finite")
    # Declared fallback only for rows whose predicted roots are all nonpositive.
    predicted[predicted.sum(axis=1) == 0.] = masses.mean(axis=0)
    predicted /= predicted.sum(axis=1, keepdims=True)
    cumulative = np.minimum(np.cumsum(predicted, axis=1) * 100., 100.)
    cumulative[:, -1] = 100.
    return cumulative


def emd_median_regression(
    train_features: np.ndarray, train_curves: np.ndarray, query_features: np.ndarray,
) -> np.ndarray:
    """Independent fixed alpha=.01 conditional medians of CDF proportions."""
    features, curves, queries = _scaled_inputs(train_features, train_curves, query_features)
    if len(queries) == 0:
        return np.empty((0, 11))
    columns = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for targets in (curves[:, :10] / 100.).T:
            if np.all(targets == targets[0]):
                prediction = np.full(len(queries), targets[0])
            else:
                model = QuantileRegressor(
                    quantile=0.5, alpha=0.01, solver="highs", fit_intercept=True,
                ).fit(features, targets)
                prediction = model.predict(queries)
            columns.append(prediction)
    predicted = np.column_stack(columns)
    if not np.isfinite(predicted).all():
        raise RuntimeError("Median-regression predictions must be finite")
    predicted = np.maximum.accumulate(np.clip(predicted, 0., 1.), axis=1)
    return np.column_stack([predicted, np.ones(len(queries))]) * 100.
