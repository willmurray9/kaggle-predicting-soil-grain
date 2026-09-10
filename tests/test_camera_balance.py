import numpy as np
import pandas as pd

from soilgrain.camera_balance import balance_cameras
from soilgrain.experiments import evaluate_samples


def test_balance_cameras_gives_each_camera_equal_weight_and_preserves_splits() -> None:
    photos = pd.DataFrame({
        "split": ["train"] * 4 + ["test"],
        "sample_id": ["A"] * 5,
        "camera": ["one", "two", "two", "two", "one"],
        "path": ["a", "b", "c", "d", "e"],
        "feature_0": [0.0, 6.0, 9.0, 12.0, 100.0],
    })

    balanced = balance_cameras(photos)

    train = balanced[balanced["split"] == "train"]
    assert len(train) == 2
    assert train["feature_0"].mean() == 4.5
    assert balanced.loc[balanced["split"] == "test", "feature_0"].item() == 100.0
    repeated = pd.concat([photos, photos[photos["camera"] == "two"]], ignore_index=True)
    pd.testing.assert_frame_equal(balanced, balance_cameras(repeated))


def test_balanced_evaluation_excludes_all_views_and_keeps_sample_order() -> None:
    photos = pd.DataFrame({
        "split": ["train"] * 9,
        "sample_id": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
        "camera": ["one", "two", "two", "one", "two", "one", "two", "two", "two"],
        "feature_0": [0.0, 10.0, 10.0, 20.0, 40.0, 40.0, 80.0, 80.0, 80.0],
    })
    ids = ["C", "A", "B"]
    curves = np.column_stack([np.repeat([[60], [5], [30]], 10, axis=1), [100] * 3])
    calls = []

    def inspect_predictor(features, targets, queries):
        calls.append((features.copy(), targets.copy(), queries.copy()))
        return np.tile(targets.mean(axis=0), (len(queries), 1))

    oof, cameras = evaluate_samples(balance_cameras(photos), ids, curves, inspect_predictor)

    expected_features = np.array([[60.0], [5.0], [30.0]])
    for i in range(3):
        keep = np.arange(3) != i
        for features, targets, _queries in calls[2 * i:2 * i + 2]:
            np.testing.assert_array_equal(features, expected_features[keep])
            np.testing.assert_array_equal(targets, curves[keep])
        np.testing.assert_array_equal(calls[2 * i][2], expected_features[i:i + 1])
    np.testing.assert_allclose(oof[:, 0], [17.5, 45.0, 32.5])
    assert cameras["sample_id"].tolist() == ["C", "C", "A", "A", "B", "B"]
    np.testing.assert_array_equal(calls[1][2], [[40.0], [80.0]])
