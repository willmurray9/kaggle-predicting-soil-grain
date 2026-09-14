import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

TREE_PARAMETERS = {
    "n_estimators": 256,
    "max_depth": 3,
    "min_samples_leaf": 3,
    "max_features": 1.0,
    "criterion": "squared_error",
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": 1,
}


def extra_trees_curves(
    train_features: np.ndarray,
    train_curves: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    """Fit one fixed shallow forest jointly to the first ten curve values."""
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

    model = ExtraTreesRegressor(**TREE_PARAMETERS).fit(train_features, train_curves[:, :10])
    predictions = model.predict(query_features)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("tree predictions must be finite")
    predictions = np.maximum.accumulate(np.clip(predictions, 0.0, 100.0), axis=1)
    return np.column_stack([predictions, np.full(len(query_features), 100.0)])
