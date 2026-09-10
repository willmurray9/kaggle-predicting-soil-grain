from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.image_model import (
    leave_one_out_predictions,
    nearest_curves,
    photo_features,
    physical_crop,
    sample_features,
    spatial_photo_features,
)


def test_default_crop_preserves_existing_center_pixels(tmp_path: Path) -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (241, 321, 3), dtype=np.uint8)
    path = tmp_path / "center.png"
    Image.fromarray(pixels).save(path)
    camera = pd.Series({"ppm": 1.0, "width": 321, "height": 241})
    expected = Image.fromarray(pixels[70:170, 110:210]).resize((256, 256), Image.Resampling.LANCZOS)
    np.testing.assert_array_equal(np.asarray(physical_crop(path, camera)), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(physical_crop(path, camera, position=(0.5, 0.5))), np.asarray(expected)
    )


def test_shifted_crop_uses_calibrated_image_dimensions(tmp_path: Path) -> None:
    pixels = np.full((240, 320, 3), [0, 0, 255], dtype=np.uint8)
    pixels[120:200, 60:140] = [255, 0, 0]
    path = tmp_path / "shifted.png"
    Image.fromarray(pixels).save(path)
    camera = pd.Series({"ppm": 2.0, "width": 640, "height": 480})
    crop = physical_crop(path, camera, crop_mm=80, position=(0.25, 0.75))
    np.testing.assert_array_equal(np.asarray(crop), np.broadcast_to([255, 0, 0], (256, 256, 3)))
    features = photo_features(path, camera, crop_mm=80, position=(0.25, 0.75))
    np.testing.assert_allclose(features[:9], [1, 0, 0] * 3)
    np.testing.assert_allclose(features[9:], 0, atol=1e-12)


def test_crop_positions_follow_exif_orientation(tmp_path: Path) -> None:
    oriented = np.zeros((320, 240, 3), dtype=np.uint8)
    oriented[180:260, 40:120] = 255
    image = Image.fromarray(np.rot90(oriented))
    exif = Image.Exif()
    exif[274] = 6  # The displayed image is rotated 90 degrees clockwise.
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    camera = pd.Series({"ppm": 2.0, "width": 640, "height": 480})
    crop = physical_crop(path, camera, crop_mm=80, position=(0.25, 0.75))
    np.testing.assert_array_equal(np.asarray(crop), 255)


@pytest.mark.parametrize("position", [(-0.1, 0.5), (0.5, 1.1), (np.nan, 0.5), (0.5, np.inf), (0.5,), (0.5, 0.5, 0.5)])
def test_crop_positions_must_be_two_finite_fractions(tmp_path: Path, position) -> None:
    path = tmp_path / "soil.png"
    Image.new("RGB", (100, 100)).save(path)
    camera = pd.Series({"ppm": 1.0, "width": 100, "height": 100})
    with pytest.raises(ValueError, match="position"):
        physical_crop(path, camera, position=position)


def test_spatial_features_average_five_grayscale_patches_equally(tmp_path: Path) -> None:
    pixels = np.full((1280, 1280, 3), 17, dtype=np.uint8)
    for (left, top), value in zip(
        [(512, 512), (256, 256), (256, 768), (768, 256), (768, 768)],
        [0, 50, 100, 150, 250], strict=True,
    ):
        pixels[top:top + 256, left:left + 256] = value
    path = tmp_path / "five_patches.png"
    Image.fromarray(pixels).save(path)
    camera = pd.Series({"ppm": 1.0, "width": 1280, "height": 1280})
    features = spatial_photo_features(path, camera, crop_mm=256)
    np.testing.assert_allclose(features, [110 / 255] * 3 + [0] * 4, atol=1e-12)


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


def test_grayscale_features_ignore_channel_color_cast(tmp_path: Path) -> None:
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(256, dtype=np.uint8)
    camera = pd.Series({"ppm": 2.56, "width": 256, "height": 256})
    features = []
    for name, rgb in [("red", pixels), ("blue", pixels[:, :, ::-1])]:
        path = tmp_path / f"{name}.png"
        Image.fromarray(rgb).save(path)
        features.append(photo_features(path, camera, color_mode="gray"))
    assert features[0].shape == (7,)
    np.testing.assert_allclose(features[0], features[1])


def test_normalized_grayscale_is_invariant_to_affine_brightness(tmp_path: Path) -> None:
    pixels = np.tile(np.arange(256, dtype=np.uint8) % 64 + 30, (256, 1))
    camera = pd.Series({"ppm": 2.56, "width": 256, "height": 256})
    features = []
    for name, gray in [("dark", pixels), ("bright", 2 * pixels + 20)]:
        path = tmp_path / f"{name}.png"
        Image.fromarray(gray).save(path)
        features.append(photo_features(path, camera, color_mode="normalized_gray"))
    np.testing.assert_allclose(features[0], features[1], atol=1e-12)


def test_crop_size_changes_the_physical_area_sampled(tmp_path: Path) -> None:
    pixels = np.full((256, 256, 3), 255, dtype=np.uint8)
    pixels[64:192, 64:192] = 0
    path = tmp_path / "soil.png"
    Image.fromarray(pixels).save(path)
    camera = pd.Series({"ppm": 2.56, "width": 256, "height": 256})
    small = photo_features(path, camera, crop_mm=50, color_mode="gray")
    large = photo_features(path, camera, crop_mm=100, color_mode="gray")
    np.testing.assert_allclose(small, 0)
    assert large[1] == 1.0


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


def test_requested_neighbor_count_and_default_behavior() -> None:
    features = np.array([[0], [1], [4], [9]], dtype=float)
    curves = np.array([[0, 100], [20, 100], [60, 100], [100, 100]], dtype=float)
    query = np.array([[0.0]])
    np.testing.assert_array_equal(nearest_curves(features, curves, query, n_neighbors=1), curves[:1])
    np.testing.assert_allclose(nearest_curves(features, curves, query, n_neighbors=2), [[10, 100]])
    np.testing.assert_array_equal(
        nearest_curves(features, curves, query),
        nearest_curves(features, curves, query, n_neighbors=3),
    )
    np.testing.assert_allclose(nearest_curves(features[:2], curves[:2], query), [[10, 100]])


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_neighbor_count_requires_a_positive_integer(count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        nearest_curves(np.zeros((1, 1)), np.zeros((1, 2)), np.zeros((1, 1)), n_neighbors=count)
