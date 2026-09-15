import numpy as np
from sklearn.isotonic import isotonic_regression

from soilgrain.constants import SUPPORT_DIAMETERS


QUANTILE_PERCENTILES = (np.arange(1000) + 0.5) / 10
LOG_SUPPORT = np.log10(SUPPORT_DIAMETERS)
LOG_SUPPORT_TOLERANCE = 1e-12


def curves_to_log_quantiles(curves: np.ndarray) -> np.ndarray:
    """Represent discrete mass CDFs at 1,000 fixed midpoint percentiles."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != 11:
        raise ValueError("Curves must contain 11 cumulative percentages per row.")
    if not np.isfinite(curves).all() or np.any((curves < 0) | (curves > 100)):
        raise ValueError("Curves must be finite and within [0, 100].")
    if np.any(np.diff(curves, axis=1) < 0) or np.any(curves[:, -1] != 100):
        raise ValueError("Curves must be nondecreasing and end at 100.")
    indices = np.array([
        np.searchsorted(curve, QUANTILE_PERCENTILES, side="left") for curve in curves
    ], dtype=int).reshape(-1, 1000)
    return LOG_SUPPORT[indices]


def log_quantiles_to_curves(quantiles: np.ndarray) -> np.ndarray:
    """Count empirical mass at each support, allowing numerical log-size equality."""
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 2 or quantiles.shape[1] != 1000:
        raise ValueError("Quantiles must contain 1,000 log-diameters per row.")
    if not np.isfinite(quantiles).all() or np.any(np.diff(quantiles, axis=1) < 0):
        raise ValueError("Quantiles must be finite and nondecreasing.")
    if np.any(quantiles < LOG_SUPPORT[0]) or np.any(quantiles > LOG_SUPPORT[-1]):
        raise ValueError("Quantiles must stay within the log-diameter support.")
    curves = np.count_nonzero(
        quantiles[:, :, None] <= LOG_SUPPORT + LOG_SUPPORT_TOLERANCE, axis=1,
    ) * 0.1
    curves[:, -1] = 100.0
    return curves


def transport_ridge_curves(
    train_features: np.ndarray, train_curves: np.ndarray, query_features: np.ndarray,
) -> np.ndarray:
    """Fit alpha-10 ridge in log-quantile space and project to valid distributions."""
    train_features = np.asarray(train_features, dtype=float)
    query_features = np.asarray(query_features, dtype=float)
    if train_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("Features must be two-dimensional.")
    if train_features.shape[0] == 0 or train_features.shape[1] == 0:
        raise ValueError("Training features must not be empty.")
    if query_features.shape[1] != train_features.shape[1]:
        raise ValueError("Query features must match the training feature count.")
    if not np.isfinite(train_features).all() or not np.isfinite(query_features).all():
        raise ValueError("Features must be finite.")
    targets = curves_to_log_quantiles(train_curves)
    if len(targets) != len(train_features):
        raise ValueError("Training curves must match the training feature rows.")
    if len(query_features) == 0:
        return np.empty((0, 11))

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0
    scaled_train = (train_features - feature_mean) / feature_std
    scaled_query = (query_features - feature_mean) / feature_std
    target_mean = targets.mean(axis=0)
    coefficients = np.linalg.solve(
        scaled_train.T @ scaled_train + 10.0 * np.eye(train_features.shape[1]),
        scaled_train.T @ (targets - target_mean),
    )
    predictions = scaled_query @ coefficients + target_mean
    projected = np.vstack([
        isotonic_regression(row, y_min=LOG_SUPPORT[0], y_max=LOG_SUPPORT[-1])
        for row in predictions
    ])
    return log_quantiles_to_curves(projected)
