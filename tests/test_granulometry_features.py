import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.granulometry_features import granulometry_features, physical_granulometry_features


@pytest.mark.parametrize("side,bin_index", [(2, 0), (4, 1), (10, 2), (20, 3), (40, 4), (82, 5)])
def test_known_square_disappears_at_first_larger_aperture(side, bin_index):
    gray = np.zeros((256, 256))
    start = (256 - side) // 2
    gray[start:start + side, start:start + side] = 1.0

    actual = granulometry_features(gray)

    expected = np.zeros(6)
    expected[bin_index] = 1.0
    np.testing.assert_allclose(actual[:6], expected, atol=1e-12)


def test_equal_histograms_with_different_geometry_have_different_patterns():
    small = np.zeros((256, 256))
    for row, column in ((80, 80), (80, 160), (160, 80), (160, 160)):
        small[row:row + 4, column:column + 4] = 1.0
    large = np.zeros_like(small)
    large[120:128, 120:128] = 1.0
    np.testing.assert_array_equal(np.sort(small.ravel()), np.sort(large.ravel()))

    np.testing.assert_allclose(granulometry_features(small)[:6], [0, 1, 0, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(granulometry_features(large)[:6], [0, 0, 1, 0, 0, 0], atol=1e-12)


def test_reflection_enlarges_corner_fragment_but_counts_only_original_region():
    gray = np.zeros((256, 256))
    gray[:4, :4] = 1.0
    gray[120:124, 120:124] = 1.0

    # Reflection makes the corner fragment 7 by 7; its original area is still
    # 16 pixels, equal to the central square that disappears at aperture five.
    np.testing.assert_allclose(granulometry_features(gray)[:6], [0, 0.5, 0.5, 0, 0, 0], atol=1e-12)


def test_polarity_inversion_swaps_bright_and_dark_patterns():
    gray = np.random.default_rng(12).uniform(0.1, 0.8, (256, 256))

    original = granulometry_features(gray)
    inverted = granulometry_features(1 - gray)

    np.testing.assert_allclose(inverted, np.r_[original[6:], original[:6]], atol=1e-12)
    assert original.shape == (12,) and np.isfinite(original).all()
    assert np.all(original >= 0)
    np.testing.assert_allclose(original.reshape(2, 6).sum(axis=1), [1, 1], atol=1e-12)


def test_pattern_is_invariant_to_affine_brightness_and_quarter_turn():
    gray = np.random.default_rng(23).uniform(0.1, 0.3, (256, 256))

    expected = granulometry_features(gray)

    np.testing.assert_allclose(granulometry_features(2 * gray + 0.1), expected, atol=1e-12)
    np.testing.assert_allclose(granulometry_features(np.rot90(gray)), expected, atol=1e-12)


@pytest.mark.parametrize("level", [0.0, 0.5, 1.0, 255.0, -3.0])
def test_constant_image_returns_zero_pattern(level):
    np.testing.assert_array_equal(granulometry_features(np.full((256, 256), level)), np.zeros(12))


def test_negligible_total_loss_returns_zeros():
    gray = np.zeros((256, 256))
    gray[128, 128] = 1e-15

    np.testing.assert_array_equal(granulometry_features(gray), np.zeros(12))


def test_physical_features_use_calibrated_center_crop_and_rgb_mean(tmp_path):
    rgb = np.full((512, 512, 3), 255, dtype=np.uint8)
    center = np.zeros((256, 256, 3), dtype=np.uint8)
    center[40:44, 40:44, 0] = 255
    center[140:150, 140:150, 1] = 255
    rgb[128:384, 128:384] = center
    path = tmp_path / "calibrated.png"
    Image.fromarray(rgb).save(path)
    camera = pd.Series({"ppm": 5.12, "width": 1024, "height": 1024})

    actual = physical_granulometry_features(path, camera)

    np.testing.assert_allclose(actual[:6], [0, 16 / 116, 100 / 116, 0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("shape", [(256,), (255, 256), (256, 256, 3)])
def test_wrong_image_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="256"):
        granulometry_features(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_image_is_rejected(value):
    gray = np.zeros((256, 256))
    gray[100, 100] = value
    with pytest.raises(ValueError, match="finite"):
        granulometry_features(gray)
