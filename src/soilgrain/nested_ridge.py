import numpy as np
import pandas as pd

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.ridge import ridge_curves

RIDGE_ALPHAS = (10.0, 100.0, 1000.0)


def select_ridge_alpha(
    features: np.ndarray, curves: np.ndarray,
    *, n_components: int | None = None,
) -> tuple[float, dict[float, float]]:
    """Select a fixed-grid penalty using only these training soils."""
    features, curves = np.asarray(features), np.asarray(curves)
    if len(features) < 2:
        raise ValueError("alpha selection requires at least two soils")
    options = {} if n_components is None else {"n_components": n_components}
    scores = {}
    for alpha in RIDGE_ALPHAS:
        predictions = []
        for i in range(len(features)):
            keep = np.arange(len(features)) != i
            predictions.append(ridge_curves(
                features[keep], curves[keep], features[i:i + 1], alpha=alpha, **options,
            )[0])
        scores[alpha] = emd_score(curves, np.vstack(predictions))
    selected = min(scores, key=lambda alpha: (scores[alpha], -alpha))
    return selected, scores


def evaluate_nested_ridge(
    photos: pd.DataFrame, sample_ids: list[str], curves: np.ndarray,
    *, n_components: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Hold out each soil, selecting its penalty inside the remaining soils."""
    if len(sample_ids) < 3:
        raise ValueError("nested evaluation requires at least three soils")
    options = {} if n_components is None else {"n_components": n_components}
    curves = np.asarray(curves)
    columns = [c for c in photos if c.startswith("feature_")]
    features = photos.groupby("sample_id")[columns].mean().loc[sample_ids].to_numpy()
    oof, camera_rows, selection_rows = [], [], []
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(sample_ids)) != i
        selected, scores = select_ridge_alpha(features[keep], curves[keep], **options)
        selection_rows.extend({
            "sample_id": sample_id, "alpha": alpha, "inner_loo_emd": score,
            "selected": alpha == selected,
        } for alpha, score in scores.items())
        views = photos[photos["sample_id"] == sample_id].groupby("camera")[columns].mean()
        queries = np.vstack([features[i:i + 1], views.to_numpy()])
        predictions = ridge_curves(features[keep], curves[keep], queries, alpha=selected, **options)
        oof.append(predictions[0])
        for camera, prediction in zip(views.index, predictions[1:], strict=True):
            camera_rows.append({
                "sample_id": sample_id, "camera": camera,
                "emd": emd_score(curves[i], prediction),
                **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
            })
    return np.vstack(oof), pd.DataFrame(camera_rows), pd.DataFrame(selection_rows)
