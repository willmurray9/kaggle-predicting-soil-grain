import numpy as np
import pandas as pd

from soilgrain.experiments import evaluate_samples
from soilgrain.image_model import nearest_curves


def test_camera_queries_exclude_every_photo_of_held_out_soil() -> None:
    photos = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B", "C", "C", "D", "D", "UNLABELED"],
            "camera": ["Phone 1", "Phone 2"] * 4 + ["Phone 1"],
            "feature_0": [0, 100, 1, 101, 2, 102, 3, 103, 1e9],
        }
    )
    curves = np.column_stack([np.repeat([[0], [30], [60], [90]], 10, axis=1), [100] * 4])
    oof, camera_predictions = evaluate_samples(photos, ["D", "B", "A", "C"], curves, nearest_curves)

    # Each query, including either camera view, must average the other 3 soils.
    np.testing.assert_allclose(oof[:, 0], [60, 50, 40, 30])
    for sample_id, expected in [("D", 60), ("B", 50), ("A", 40), ("C", 30)]:
        rows = camera_predictions[camera_predictions["sample_id"] == sample_id]
        assert len(rows) == 2
        np.testing.assert_allclose(rows["0.002"], expected)
    assert "UNLABELED" not in camera_predictions["sample_id"].values
