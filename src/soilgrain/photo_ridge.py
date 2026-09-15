import numpy as np
import pandas as pd

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.targets import validate_cumulative_curves


def _training_data(photos, sample_ids, curves):
    ids = pd.Index(sample_ids)
    if len(ids) == 0 or not ids.is_unique or ids.hasnans:
        raise ValueError("Training soil IDs must be nonempty, unique, and non-null.")
    if not photos.columns.is_unique or "sample_id" not in photos:
        raise ValueError("Photos require unique columns and sample_id keys.")
    if photos["sample_id"].isna().any() or set(photos["sample_id"]) != set(ids):
        raise ValueError("Photo soil coverage must exactly match training soil IDs.")
    columns = [c for c in photos if isinstance(c, str) and c.startswith("feature_")]
    if not columns:
        raise ValueError("Photos must contain feature columns.")
    features = photos[columns].to_numpy(dtype=float)
    curves = np.asarray(curves, dtype=float)
    if curves.shape != (len(ids), 11):
        raise ValueError("Training curves must contain one 11-value curve per soil ID.")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(curves)):
        raise ValueError("Features and curves must be finite.")
    validate_cumulative_curves(pd.DataFrame(curves, columns=CANONICAL_GRAIN_LABELS))
    means = photos.groupby("sample_id")[columns].mean().loc[sample_ids].to_numpy(dtype=float)
    return columns, features, curves, means


def photo_ridge_curves(
    train_photos: pd.DataFrame,
    train_ids: list[str],
    train_curves: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    """Fit alpha-10 ridge on photos, giving each physical soil total weight one."""
    columns, features, curves, means = _training_data(train_photos, train_ids, train_curves)
    query_features = np.asarray(query_features, dtype=float)
    if query_features.ndim != 2 or query_features.shape[1] != len(columns):
        raise ValueError("Query features must match the training feature count.")
    if not np.all(np.isfinite(query_features)):
        raise ValueError("Query features must be finite.")

    feature_mean = means.mean(axis=0)
    feature_std = means.std(axis=0)
    feature_std[feature_std == 0.0] = 1.0
    scaled_train = (features - feature_mean) / feature_std
    scaled_query = (query_features - feature_mean) / feature_std
    positions = pd.Index(train_ids).get_indexer(train_photos["sample_id"])
    weights = 1.0 / np.bincount(positions)[positions]
    target_mean = curves[:, :10].mean(axis=0)
    centered_targets = curves[positions, :10] - target_mean
    coefficients = np.linalg.solve(
        scaled_train.T @ (weights[:, None] * scaled_train) + 10.0 * np.eye(len(columns)),
        scaled_train.T @ (weights[:, None] * centered_targets),
    )
    predictions = scaled_query @ coefficients + target_mean
    predictions = np.maximum.accumulate(np.clip(predictions, 0.0, 100.0), axis=1)
    return np.column_stack([predictions, np.full(len(query_features), 100.0)])


def evaluate_photo_ridge(
    photos: pd.DataFrame, sample_ids: list[str], curves: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Exclude every photo of one soil and query its pooled and camera means together."""
    columns, _, curves, means = _training_data(photos, sample_ids, curves)
    if len(sample_ids) < 2:
        raise ValueError("Whole-soil validation requires at least two soils.")
    if "camera" not in photos or photos["camera"].isna().any():
        raise ValueError("Photos require non-null camera keys.")
    oof, camera_rows = [], []
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(sample_ids)) != i
        views = photos[photos["sample_id"] == sample_id].groupby("camera")[columns].mean()
        queries = np.vstack([means[i:i + 1], views.to_numpy(dtype=float)])
        predictions = photo_ridge_curves(
            photos[photos["sample_id"] != sample_id],
            [soil for soil, included in zip(sample_ids, keep, strict=True) if included],
            curves[keep], queries,
        )
        oof.append(predictions[0])
        for camera, prediction in zip(views.index, predictions[1:], strict=True):
            camera_rows.append({
                "sample_id": sample_id, "camera": camera,
                "emd": emd_score(curves[i], prediction),
                **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
            })
    return np.vstack(oof), pd.DataFrame(camera_rows)
