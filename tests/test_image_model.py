from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from soilgrain.image_model import leave_one_out_predictions, nearest_curves, photo_features, sample_features


def test_photo_features_use_physical_center_crop(tmp_path: Path) -> None:
    # Both cameras see a 100 mm red square; surrounding blue must be excluded.
    features = []
    for size, ppm in [(300, 1.0), (600, 2.0)]:
        pixels = np.zeros((size, size, 3), dtype=np.uint8)
        pixels[:, :, 2] = 255
        start, stop = size // 3, 2 * size // 3
        pixels[start:stop, start:stop] = [255, 0, 0]
        path = tmp_path / f"{size}.png"
        Image.fromarray(pixels).save(path)
        camera = pd.Series({"ppm": ppm, "width": size, "height": size})
        features.append(photo_features(path, camera))
    np.testing.assert_allclose(features[0], features[1], atol=1e-6)
    np.testing.assert_allclose(features[0][:9], [1, 0, 0] * 3)
    np.testing.assert_allclose(features[0][9:], 0, atol=1e-6)


def test_photo_features_correct_for_downsampled_images(tmp_path: Path) -> None:
    pixels = np.zeros((300, 300, 3), dtype=np.uint8)
    pixels[100:200, 100:200] = 255
    path = tmp_path / "resized.png"
    Image.fromarray(pixels).save(path)
    camera = pd.Series({"ppm": 2.0, "width": 600, "height": 600})
    np.testing.assert_allclose(photo_features(path, camera)[:9], 1)


def test_texture_features_distinguish_equal_color_distributions(tmp_path: Path) -> None:
    checkerboard = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(np.uint8)
    halves = np.zeros((256, 256), dtype=np.uint8)
    halves[:, 128:] = 255
    camera = pd.Series({"ppm": 2.56, "width": 256, "height": 256})
    features = []
    for name, pixels in [("checkerboard", checkerboard), ("halves", halves)]:
        path = tmp_path / f"{name}.png"
        Image.fromarray(pixels).save(path)
        features.append(photo_features(path, camera))
    np.testing.assert_allclose(features[0][:10], features[1][:10])
    assert features[0][10] > 0.9
    assert features[1][10] < 0.01


def test_sample_features_aggregate_photos_in_requested_id_order(tmp_path: Path) -> None:
    rows = []
    for name, sample_id, color in [("a", "A", "black"), ("b", "A", "white"), ("c", "B", "red")]:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (100, 100), color).save(path)
        rows.append({"sample_id": sample_id, "path": str(path), "camera": "Phone"})
    ppm = pd.DataFrame({"phone": ["Phone"], "width": [100], "height": [100], "ppm": [1.0]})
    values = sample_features(pd.DataFrame(rows), ["B", "A"], ppm)
    np.testing.assert_allclose(values[0, :9], [1, 0, 0] * 3)
    np.testing.assert_allclose(values[1, :9], 0.5)


def test_leave_one_out_excludes_entire_held_out_sample() -> None:
    # With four physical samples, each prediction must average the other three.
    features = np.arange(4, dtype=float).reshape(-1, 1)
    curves = np.array([[0, 100], [30, 100], [60, 100], [90, 100]], dtype=float)
    predictions = leave_one_out_predictions(features, curves)
    np.testing.assert_allclose(predictions, [[60, 100], [50, 100], [40, 100], [30, 100]])


def test_neighbors_do_not_fit_scaling_on_query_samples() -> None:
    features = np.array([[0, 0], [1, 1], [2, 2], [100, 3]], dtype=float)
    curves = np.array([[0, 100], [30, 100], [60, 100], [90, 100]], dtype=float)
    alone = nearest_curves(features, curves, np.array([[0.0, 0.0]]))
    together = nearest_curves(features, curves, np.array([[0.0, 0.0], [1e9, 0.0]]))
    np.testing.assert_allclose(alone[0], [30, 100])
    np.testing.assert_allclose(alone[0], together[0])
