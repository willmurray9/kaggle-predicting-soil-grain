import numpy as np
import pandas as pd

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.kernel_ridge import kernel_ridge_curves
from soilgrain.metrics import emd_score

KERNEL_SETTINGS = tuple((alpha, scale) for alpha in (0.1, 1.0, 10.0) for scale in (0.1, 1.0))


def select_kernel_parameters(
    features: np.ndarray, curves: np.ndarray,
) -> tuple[tuple[float, float], dict[tuple[float, float], float]]:
    """Select a fixed-grid penalty and bandwidth using only these training soils."""
    features, curves = np.asarray(features), np.asarray(curves)
    if len(features) < 2:
        raise ValueError("kernel selection requires at least two soils")
    scores = {}
    for alpha, gamma_scale in KERNEL_SETTINGS:
        predictions = []
        for i in range(len(features)):
            keep = np.arange(len(features)) != i
            predictions.append(kernel_ridge_curves(
                features[keep], curves[keep], features[i:i + 1],
                alpha=alpha, gamma=gamma_scale / features.shape[1],
            )[0])
        scores[(alpha, gamma_scale)] = emd_score(curves, np.vstack(predictions))
    selected = min(scores, key=lambda pair: (scores[pair], -pair[0], pair[1]))
    return selected, scores


def evaluate_nested_kernel(
    photos: pd.DataFrame, sample_ids: list[str], curves: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Hold out each soil, selecting its kernel settings inside the remaining soils."""
    if len(sample_ids) < 3:
        raise ValueError("nested evaluation requires at least three soils")
    curves = np.asarray(curves)
    columns = [c for c in photos if c.startswith("feature_")]
    features = photos.groupby("sample_id")[columns].mean().loc[sample_ids].to_numpy()
    oof, camera_rows, selection_rows = [], [], []
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(sample_ids)) != i
        selected, scores = select_kernel_parameters(features[keep], curves[keep])
        selection_rows.extend({
            "sample_id": sample_id, "alpha": alpha, "gamma_scale": gamma_scale,
            "inner_loo_emd": score, "selected": (alpha, gamma_scale) == selected,
        } for (alpha, gamma_scale), score in scores.items())
        views = photos[photos["sample_id"] == sample_id].groupby("camera")[columns].mean()
        queries = np.vstack([features[i:i + 1], views.to_numpy()])
        predictions = kernel_ridge_curves(
            features[keep], curves[keep], queries,
            alpha=selected[0], gamma=selected[1] / features.shape[1],
        )
        oof.append(predictions[0])
        for camera, prediction in zip(views.index, predictions[1:], strict=True):
            camera_rows.append({
                "sample_id": sample_id, "camera": camera,
                "emd": emd_score(curves[i], prediction),
                **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
            })
    return np.vstack(oof), pd.DataFrame(camera_rows), pd.DataFrame(selection_rows)
