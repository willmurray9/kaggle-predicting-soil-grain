import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR

SVR_EPSILON = 1.0
SVR_TOL = 1e-6
SVR_MAX_ITER = 1_000_000


def svr_curves(
    train_features: np.ndarray,
    train_curves: np.ndarray,
    query_features: np.ndarray,
    *,
    C: float = 1.0,
) -> np.ndarray:
    """Fit linear epsilon-SVR to each diameter using training-only scaling."""
    train_features = np.asarray(train_features, dtype=float)
    train_curves = np.asarray(train_curves, dtype=float)
    query_features = np.asarray(query_features, dtype=float)
    if train_features.ndim != 2 or query_features.ndim != 2 or train_curves.ndim != 2:
        raise ValueError("features and curves must be two-dimensional")
    if train_features.shape[0] == 0 or train_features.shape[1] == 0:
        raise ValueError("training features must not be empty")
    if train_curves.shape != (train_features.shape[0], 11):
        raise ValueError("train_curves must have one 11-value curve per training row")
    if query_features.shape[1] != train_features.shape[1]:
        raise ValueError("query features must match the training feature count")
    if not all(np.all(np.isfinite(values)) for values in (train_features, train_curves, query_features)):
        raise ValueError("features and curves must be finite")
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive and finite")
    if len(query_features) == 0:
        return np.empty((0, 11))

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0
    scaled_train = (train_features - feature_mean) / feature_std
    scaled_query = (query_features - feature_mean) / feature_std
    columns = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for targets in train_curves[:, :10].T:
            model = SVR(
                kernel="linear", C=C, epsilon=SVR_EPSILON, tol=SVR_TOL,
                max_iter=SVR_MAX_ITER, shrinking=True,
            ).fit(scaled_train, targets)
            if model.fit_status_ != 0:
                raise RuntimeError("SVR did not converge")
            columns.append(model.predict(scaled_query))
    predictions = np.column_stack(columns)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("SVR predictions must be finite")
    predictions = np.maximum.accumulate(np.clip(predictions, 0.0, 100.0), axis=1)
    return np.column_stack([predictions, np.full(len(query_features), 100.0)])
