import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score

PLS_COMPONENTS = (1, 2, 4)


def pls_curves(
    train_features: np.ndarray, train_curves: np.ndarray, query_features: np.ndarray,
    *, n_components: int,
) -> np.ndarray:
    """Fit supervised PLS with training-only feature scaling and raw CDF targets."""
    train_features, train_curves, query_features = [
        np.asarray(values, dtype=float) for values in (train_features, train_curves, query_features)
    ]
    if any(values.ndim != 2 for values in (train_features, train_curves, query_features)):
        raise ValueError("Features and curves must be two-dimensional")
    if min(train_features.shape) == 0 or train_curves.shape != (len(train_features), 11):
        raise ValueError("Require nonempty training features and one 11-value curve per row")
    if query_features.shape[1] != train_features.shape[1]:
        raise ValueError("Query features must match the training feature count")
    if not all(np.isfinite(values).all() for values in (train_features, train_curves, query_features)):
        raise ValueError("Features and curves must be finite")
    if (isinstance(n_components, (bool, np.bool_)) or not isinstance(n_components, (int, np.integer))
            or not 1 <= n_components <= min(train_features.shape[1], len(train_features) - 1)):
        raise ValueError("n_components must be between 1 and min(feature count, training rows - 1)")
    if len(query_features) == 0:
        return np.empty((0, 11))

    with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise", divide="raise"):
        warnings.simplefilter("error", ConvergenceWarning)
        mean, scale = train_features.mean(axis=0), train_features.std(axis=0)
        scale[scale == 0] = 1
        scaled_train, scaled_query = (train_features - mean) / scale, (query_features - mean) / scale
        targets = train_curves[:, :10]
        if np.all(train_features == train_features[0]) or np.all(targets == targets[0]):
            predictions = np.tile(targets.mean(axis=0), (len(query_features), 1))
        else:
            model = PLSRegression(n_components=n_components, scale=False, max_iter=500, tol=1e-6, copy=True)
            model.fit(scaled_train, targets)
            predictions = model.predict(scaled_query)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("PLS predictions must be finite")
    predictions = np.maximum.accumulate(np.clip(predictions, 0, 100), axis=1)
    return np.column_stack([predictions, np.full(len(query_features), 100)])


def select_pls_components(features: np.ndarray, curves: np.ndarray) -> tuple[int, dict[int, float]]:
    """Choose one shared component count using only the supplied training soils."""
    features, curves = np.asarray(features), np.asarray(curves)
    if features.ndim != 2 or len(features) < 6 or features.shape[1] < 4:
        raise ValueError("PLS selection requires at least six soils and four features")
    scores = {}
    for components in PLS_COMPONENTS:
        predictions = []
        for i in range(len(features)):
            keep = np.arange(len(features)) != i
            predictions.append(pls_curves(
                features[keep], curves[keep], features[i:i + 1], n_components=components,
            )[0])
        scores[components] = emd_score(curves, np.vstack(predictions))
    selected = min(scores, key=lambda k: (scores[k], k))
    return selected, scores


def evaluate_nested_pls(
    photos: pd.DataFrame, sample_ids: list[str], curves: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Exclude every held-out soil photo from component learning and selection."""
    ids = pd.Index(sample_ids)
    curves = np.asarray(curves, dtype=float)
    columns = [c for c in photos if c.startswith("feature_")]
    if len(ids) < 7 or ids.has_duplicates or ids.isna().any() or len(columns) < 4:
        raise ValueError("Nested PLS requires at least seven unique soils and four features")
    if curves.shape != (len(ids), 11) or not np.isfinite(curves).all():
        raise ValueError("Require one finite 11-value curve per soil")
    if (not {"sample_id", "camera"} <= set(photos)
            or photos[["sample_id", "camera"]].isna().any().any()
            or set(photos.sample_id) != set(ids)
            or ("split" in photos and not photos["split"].eq("train").all())
            or not np.isfinite(photos[columns].to_numpy(dtype=float)).all()):
        raise ValueError("Photos require complete training-only soil/camera keys and finite features")
    with np.errstate(over="raise", invalid="raise"):
        photos = photos.astype({c: np.float32 for c in columns})
    features = photos.groupby("sample_id")[columns].mean().loc[sample_ids].to_numpy()
    oof, camera_rows, selection_rows = [], [], []
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(sample_ids)) != i
        selected, scores = select_pls_components(features[keep], curves[keep])
        selection_rows.extend({
            "sample_id": sample_id, "n_components": k, "inner_loo_emd": score, "selected": k == selected,
        } for k, score in scores.items())
        views = photos[photos.sample_id == sample_id].groupby("camera")[columns].mean()
        queries = np.vstack([features[i:i + 1], views.to_numpy()])
        predictions = pls_curves(features[keep], curves[keep], queries, n_components=selected)
        oof.append(predictions[0])
        for camera, prediction in zip(views.index, predictions[1:], strict=True):
            camera_rows.append({
                "sample_id": sample_id, "camera": camera, "emd": emd_score(curves[i], prediction),
                **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
            })
    return np.vstack(oof), pd.DataFrame(camera_rows), pd.DataFrame(selection_rows)
