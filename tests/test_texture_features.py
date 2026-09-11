from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.texture_features import physical_texture_features


@pytest.fixture
def camera() -> pd.Series:
    return pd.Series({"ppm": 2.56, "width": 256, "height": 256})


def test_checkerboard_has_known_normalized_texture(tmp_path: Path, camera: pd.Series) -> None:
    pixels = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(np.uint8)
    path = tmp_path / "checkerboard.png"
    Image.fromarray(pixels).save(path)

    np.testing.assert_allclose(physical_texture_features(path, camera), [2, 2, 0, 0])


def test_step_counts_only_valid_pixel_pairs(tmp_path: Path, camera: pd.Series) -> None:
    pixels = np.zeros((256, 256), dtype=np.uint8)
    pixels[:, 128:] = 255
    path = tmp_path / "step.png"
    Image.fromarray(pixels).save(path)

    np.testing.assert_allclose(physical_texture_features(path, camera), [3 / 253, 5 / 251, 10 / 246, 20 / 236])


def test_constant_crop_has_zero_texture(tmp_path: Path, camera: pd.Series) -> None:
    path = tmp_path / "constant.png"
    Image.new("RGB", (256, 256), (20, 80, 140)).save(path)

    np.testing.assert_array_equal(physical_texture_features(path, camera), np.zeros(4))


def test_texture_is_invariant_to_unclipped_affine_brightness(tmp_path: Path, camera: pd.Series) -> None:
    pixels = np.random.default_rng(4).integers(30, 94, (256, 256, 3), dtype=np.uint8)
    features = []
    for name, rgb in [("dark", pixels), ("bright", 2 * pixels + 20)]:
        path = tmp_path / f"{name}.png"
        Image.fromarray(rgb).save(path)
        features.append(physical_texture_features(path, camera))

    np.testing.assert_allclose(features[0], features[1], atol=1e-12)


def test_texture_is_invariant_to_quarter_turn(tmp_path: Path, camera: pd.Series) -> None:
    pixels = np.zeros((256, 256), dtype=np.uint8)
    pixels[40:130, 90:240] = 200
    pixels[170:200, 20:100] = 80
    features = []
    for name, gray in [("original", pixels), ("rotated", np.rot90(pixels))]:
        path = tmp_path / f"{name}.png"
        Image.fromarray(gray).save(path)
        features.append(physical_texture_features(path, camera))

    np.testing.assert_allclose(features[0], features[1], atol=1e-12)
