import numpy as np
import pandas as pd

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.image_model import nearest_curves
from soilgrain.metrics import emd_score
from soilgrain.multicrop import COMPONENTS

NEIGHBOR_COUNTS = (1, 3, 5, 7)


def _check_components(components: dict) -> None:
    if set(components) != set(COMPONENTS):
        raise ValueError(f"Expected exactly these components: {list(COMPONENTS)}")


def predict_multicrop_neighbors(
    features: dict[str, np.ndarray], curves: np.ndarray,
    queries: dict[str, np.ndarray], *, n_neighbors: int,
) -> np.ndarray:
    """Use one neighbor count for the equal blend of three aligned crop models."""
    _check_components(features)
    _check_components(queries)
    predictions = []
    query_rows = len(queries[COMPONENTS[0]])
    for name in COMPONENTS:
        train, query = np.asarray(features[name]), np.asarray(queries[name])
        if (train.ndim != 2 or query.ndim != 2 or train.shape[0] != len(curves)
                or query.shape != (query_rows, train.shape[1]) or train.shape[1] == 0):
            raise ValueError(f"Invalid aligned feature shapes for {name}")
        if not np.isfinite(train).all() or not np.isfinite(query).all():
            raise ValueError(f"Non-finite features for {name}")
        predictions.append(nearest_curves(train, curves, query, n_neighbors=n_neighbors))
    return np.mean(predictions, axis=0)


def select_neighbor_count(
    features: dict[str, np.ndarray], curves: np.ndarray,
) -> tuple[int, dict[int, float]]:
    """Select a shared count by blended LOO error using only these training soils."""
    _check_components(features)
    curves = np.asarray(curves)
    features = {name: np.asarray(features[name]) for name in COMPONENTS}
    if len(curves) < 8:
        raise ValueError("neighbor selection requires at least eight soils")
    if any(len(values) != len(curves) for values in features.values()):
        raise ValueError("Feature rows must match the training curves")
    scores = {}
    for count in NEIGHBOR_COUNTS:
        predictions = []
        for i in range(len(curves)):
            keep = np.arange(len(curves)) != i
            predictions.append(predict_multicrop_neighbors(
                {name: values[keep] for name, values in features.items()}, curves[keep],
                {name: values[i:i + 1] for name, values in features.items()}, n_neighbors=count,
            )[0])
        scores[count] = emd_score(curves, np.vstack(predictions))
    selected = min(scores, key=lambda count: (scores[count], -count))
    return selected, scores


def evaluate_nested_neighbors(
    photos: dict[str, pd.DataFrame], sample_ids: list[str], curves: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Hold out each entire soil, selecting one count inside its training fold."""
    _check_components(photos)
    if len(sample_ids) < 9:
        raise ValueError("nested neighbor evaluation requires at least nine soils")
    curves = np.asarray(curves)
    if len(set(sample_ids)) != len(sample_ids) or len(curves) != len(sample_ids):
        raise ValueError("Unique sample IDs must align with the training curves")
    features, camera_features = {}, {}
    camera_index = None
    for name in COMPONENTS:
        frame = photos[name]
        if frame[["sample_id", "camera"]].isna().any().any():
            raise ValueError(f"Null soil or camera keys for {name}")
        if set(frame["sample_id"]) != set(sample_ids):
            raise ValueError(f"Soil coverage mismatch for {name}")
        columns = [column for column in frame if column.startswith("feature_")]
        features[name] = frame.groupby("sample_id")[columns].mean().loc[sample_ids].to_numpy()
        views = frame.groupby(["sample_id", "camera"])[columns].mean()
        if camera_index is None:
            camera_index = views.index
        elif set(views.index) != set(camera_index):
            raise ValueError(f"Soil/camera key coverage mismatch for {name}")
        camera_features[name] = views.reindex(camera_index)

    oof, camera_rows, selection_rows = [], [], []
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(sample_ids)) != i
        training = {name: values[keep] for name, values in features.items()}
        selected, scores = select_neighbor_count(training, curves[keep])
        selection_rows.extend({
            "sample_id": sample_id, "n_neighbors": count, "inner_loo_emd": score,
            "selected": count == selected,
        } for count, score in scores.items())
        views = {name: values.loc[sample_id] for name, values in camera_features.items()}
        queries = {
            name: np.vstack([features[name][i:i + 1], views[name].to_numpy()])
            for name in COMPONENTS
        }
        predictions = predict_multicrop_neighbors(training, curves[keep], queries, n_neighbors=selected)
        oof.append(predictions[0])
        for camera, prediction in zip(views[COMPONENTS[0]].index, predictions[1:], strict=True):
            camera_rows.append({
                "sample_id": sample_id, "camera": camera,
                "emd": emd_score(curves[i], prediction),
                **dict(zip(CANONICAL_GRAIN_LABELS, prediction, strict=True)),
            })
    return np.vstack(oof), pd.DataFrame(camera_rows), pd.DataFrame(selection_rows)
