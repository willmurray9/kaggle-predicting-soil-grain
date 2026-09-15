from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.spectral_features import physical_spectrum_features, spectrum_features


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0, 255.0, -3.0, 1e-9])
def test_constant_gray_has_zero_spectrum(level: float) -> None:
    np.testing.assert_array_equal(spectrum_features(np.full((256, 256), level)), np.zeros(6))


@pytest.mark.parametrize("band,cycles", list(enumerate([75, 36, 18, 9, 4, 2])))
def test_known_spatial_frequency_dominates_its_band(band: int, cycles: int) -> None:
    gray = np.tile(0.5 + 0.2 * np.sin(2 * np.pi * cycles * np.arange(256) / 256), (256, 1))

    features = spectrum_features(gray)

    assert features.shape == (6,)
    assert np.all(features >= 0)
    assert features.sum() == pytest.approx(1.0)
    assert np.argmax(features) == band
    assert features[band] > 0.8


def test_spectrum_is_invariant_to_unclipped_affine_brightness() -> None:
    gray = np.random.default_rng(3).uniform(0.1, 0.3, (256, 256))

    np.testing.assert_allclose(spectrum_features(gray), spectrum_features(2 * gray + 0.1), atol=1e-12)


def test_spectrum_is_invariant_to_quarter_turn() -> None:
    gray = np.zeros((256, 256))
    gray[30:110, 50:240] = 0.8
    gray[150:220, 40:65] = 0.2

    np.testing.assert_allclose(spectrum_features(gray), spectrum_features(np.rot90(gray)), atol=1e-12)


def test_negligible_power_returns_zeros() -> None:
    gray = 0.5 + np.random.default_rng(1).normal(0, 1e-12, (256, 256))

    np.testing.assert_array_equal(spectrum_features(gray), np.zeros(6))


def test_physical_spectrum_uses_calibrated_center_crop_and_rgb_mean(tmp_path: Path) -> None:
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    center = np.random.default_rng(8).integers(20, 200, (256, 256, 3), dtype=np.uint8)
    image[128:384, 128:384] = center
    path = tmp_path / "calibrated.png"
    Image.fromarray(image).save(path)
    camera = pd.Series({"ppm": 5.12, "width": 1024, "height": 1024})

    expected = spectrum_features((center.astype(float) / 255).mean(axis=2))

    np.testing.assert_allclose(physical_spectrum_features(path, camera), expected, atol=1e-12)


@pytest.mark.parametrize("shape", [(256,), (255, 256), (256, 256, 3)])
def test_spectrum_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="256"):
        spectrum_features(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_spectrum_rejects_nonfinite_gray(value: float) -> None:
    gray = np.zeros((256, 256))
    gray[100, 100] = value

    with pytest.raises(ValueError, match="finite"):
        spectrum_features(gray)
