import numpy as np


def kernel_ridge_curves(
    train_features: np.ndarray,
    train_curves: np.ndarray,
    query_features: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit RBF kernel ridge and predict valid cumulative grain curves."""
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

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0
    scaled_train = (train_features - feature_mean) / feature_std
    scaled_query = (query_features - feature_mean) / feature_std

    gamma = 1.0 / train_features.shape[1]
    kernel = np.exp(-gamma * np.sum((scaled_train[:, None] - scaled_train[None, :]) ** 2, axis=2))
    query_kernel = np.exp(-gamma * np.sum((scaled_query[:, None] - scaled_train[None, :]) ** 2, axis=2))
    kernel_mean = kernel.mean(axis=0)
    grand_mean = kernel_mean.mean()
    centered_kernel = kernel - kernel.mean(axis=1, keepdims=True) - kernel_mean + grand_mean
    centered_query = query_kernel - query_kernel.mean(axis=1, keepdims=True) - kernel_mean + grand_mean

    targets = train_curves[:, :10]
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    coefficients = np.linalg.solve(
        centered_kernel + alpha * np.eye(train_features.shape[0]), centered_targets
    )
    predictions = centered_query @ coefficients + target_mean
    predictions = np.maximum.accumulate(np.clip(predictions, 0.0, 100.0), axis=1)
    return np.column_stack([predictions, np.full(query_features.shape[0], 100.0)])
