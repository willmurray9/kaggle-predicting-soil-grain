import numpy as np


def ridge_curves(
    train_features: np.ndarray,
    train_curves: np.ndarray,
    query_features: np.ndarray,
    *,
    alpha: float = 10.0,
    n_components: int | None = None,
) -> np.ndarray:
    """Fit ridge with optional training-only PCA and predict cumulative curves."""
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
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    if n_components is not None and (
        isinstance(n_components, (bool, np.bool_))
        or not isinstance(n_components, (int, np.integer))
        or not 1 <= n_components <= min(train_features.shape[1], train_features.shape[0] - 1)
    ):
        raise ValueError("n_components must be an integer between 1 and min(feature count, training rows - 1)")

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0
    scaled_train = (train_features - feature_mean) / feature_std
    scaled_query = (query_features - feature_mean) / feature_std
    if n_components is not None:
        _, _, directions = np.linalg.svd(scaled_train, full_matrices=False)
        basis = directions[:n_components].T
        scaled_train = scaled_train @ basis
        scaled_query = scaled_query @ basis

    targets = train_curves[:, :10]
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    if scaled_train.shape[1] > scaled_train.shape[0]:
        coefficients = scaled_train.T @ np.linalg.solve(
            scaled_train @ scaled_train.T + alpha * np.eye(scaled_train.shape[0]),
            centered_targets,
        )
    else:
        coefficients = np.linalg.solve(
            scaled_train.T @ scaled_train + alpha * np.eye(scaled_train.shape[1]),
            scaled_train.T @ centered_targets,
        )
    predictions = scaled_query @ coefficients + target_mean
    predictions = np.maximum.accumulate(np.clip(predictions, 0.0, 100.0), axis=1)
    return np.column_stack([predictions, np.full(query_features.shape[0], 100.0)])
