import numpy as np
import pandas as pd

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.svr import svr_curves

SVR_C_VALUES = (0.1, 1.0, 10.0)


def select_svr_c(
    features: np.ndarray, curves: np.ndarray,
) -> tuple[float, dict[float, float]]:
    """Select one penalty for every diameter using only these training soils."""
    features, curves = np.asarray(features), np.asarray(curves)
    if len(features) < 2:
        raise ValueError("SVR selection requires at least two soils")
    scores = {}
    for C in SVR_C_VALUES:
        predictions = []
        for i in range(len(features)):
            keep = np.arange(len(features)) != i
            predictions.append(svr_curves(
                features[keep], curves[keep], features[i:i + 1], C=C,
            )[0])
        scores[C] = emd_score(curves, np.vstack(predictions))
    selected = min(scores, key=lambda C: (scores[C], C))
    return selected, scores


def evaluate_nested_svr(
    photos: pd.DataFrame, sample_ids: list[str], curves: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Hold out each soil, selecting its penalty inside the remaining soils."""
    if len(sample_ids) < 3:
        raise ValueError("nested evaluation requires at least three soils")
    curves = np.asarray(curves)
    columns = [c for c in photos if c.startswith("feature_")]
    features = photos.groupby("sample_id")[columns].mean().loc[sample_ids].to_numpy()
    oof, camera_rows, selection_rows = [], [], []
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(sample_ids)) != i
        selected, scores = select_svr_c(features[keep], curves[keep])
        selection_rows.extend({
            "sample_id": sample_id, "C": C, "inner_loo_emd": score,
            "selected": C == selected,
        } for C, score in scores.items())
        views = photos[photos["sample_id"] == sample_id].groupby("camera")[columns].mean()
        queries = np.vstack([features[i:i + 1], views.to_numpy()])
        predictions = svr_curves(features[keep], curves[keep], queries, C=selected)
        oof.append(predictions[0])
        for camera, prediction in zip(views.index, predictions[1:], strict=True):
            camera_rows.append({
                "sample_id": sample_id, "camera": camera,
                "emd": emd_score(curves[i], prediction),
                **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
            })
    return np.vstack(oof), pd.DataFrame(camera_rows), pd.DataFrame(selection_rows)
