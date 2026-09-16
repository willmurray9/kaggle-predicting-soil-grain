import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

BOOST_PARAMETERS = {
    "n_estimators": 128,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_samples_leaf": 4,
    "loss": "squared_error",
    "max_features": 1.0,
    "subsample": 1.0,
    "random_state": 42,
    "n_iter_no_change": None,
}


def boost_curves(
    train_features: np.ndarray,
    train_curves: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    """Fit ten fixed shallow boosted regressors on training-soil feature means."""
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
    if len(query_features) == 0:
        return np.empty((0, 11))

    predictions = np.column_stack([
        GradientBoostingRegressor(**BOOST_PARAMETERS)
        .fit(train_features, train_curves[:, column])
        .predict(query_features)
        for column in range(10)
    ])
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("boost predictions must be finite")
    predictions = np.maximum.accumulate(np.clip(predictions, 0.0, 100.0), axis=1)
    return np.column_stack([predictions, np.full(len(query_features), 100.0)])
