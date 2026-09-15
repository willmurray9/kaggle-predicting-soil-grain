import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.patch_features import patch_iqr_features, physical_patch_iqr_features


def test_two_colors_have_known_rgb_spread_without_boundary_texture():
    rgb = np.empty((256, 256, 3))
    rgb[:128] = [0.1, 0.3, 0.8]
    rgb[128:] = [0.9, 0.5, 0.2]

    np.testing.assert_allclose(patch_iqr_features(rgb), [0.8, 0.2, 0.6, 0, 0, 0], atol=1e-12)


def test_ramp_texture_has_analytic_spread_at_unresized_lags():
    rgb = np.full((256, 256, 3), 0.5)
    ramp = np.tile(np.arange(64) / 63, (64, 1))
    rgb[:128] = np.tile(ramp, (2, 4))[..., None]

    expected = np.r_[np.zeros(3), np.array([1, 4, 16]) / 126]
    np.testing.assert_allclose(patch_iqr_features(rgb), expected, atol=1e-12)


def test_equal_histograms_reveal_different_regional_organization():
    regions = np.zeros((256, 256, 3))
    regions[128:] = 1
    checker = np.indices((256, 256)).sum(axis=0) % 2
    mixed = np.repeat(checker[..., None], 3, axis=2)
    np.testing.assert_array_equal(np.sort(regions.ravel()), np.sort(mixed.ravel()))

    np.testing.assert_array_equal(patch_iqr_features(regions), [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(patch_iqr_features(mixed), np.zeros(6))


def test_identical_textured_tiles_have_no_spread():
    tile = np.random.default_rng(8).uniform(0, 1, (64, 64, 3))

    np.testing.assert_array_equal(patch_iqr_features(np.tile(tile, (4, 4, 1))), np.zeros(6))


@pytest.mark.parametrize("count,spread", [(0, 0), (3, 0), (4, 0.25), (8, 1), (16, 0)])
def test_linear_iqr_ignores_sparse_outliers_and_interpolates_at_quartile(count, spread):
    rgb = np.zeros((256, 256, 3))
    for index in range(count):
        row, column = divmod(index, 4)
        rgb[64 * row:64 * (row + 1), 64 * column:64 * (column + 1)] = 1

    np.testing.assert_array_equal(patch_iqr_features(rgb), [spread, spread, spread, 0, 0, 0])


def test_tile_permutation_preserves_spread():
    rgb = np.random.default_rng(11).uniform(0, 1, (256, 256, 3))
    reordered = np.concatenate([rgb[192:], rgb[64:128], rgb[:64], rgb[128:192]], axis=0)
    reordered = np.concatenate([reordered[:, 128:], reordered[:, :128]], axis=1)

    np.testing.assert_allclose(patch_iqr_features(reordered), patch_iqr_features(rgb), atol=1e-12)


def test_quarter_turn_and_reflections_preserve_spread():
    rgb = np.random.default_rng(4).uniform(0, 1, (256, 256, 3))
    expected = patch_iqr_features(rgb)

    for transformed in (np.rot90(rgb), rgb[::-1], rgb[:, ::-1]):
        np.testing.assert_allclose(patch_iqr_features(transformed), expected, atol=1e-12)


def test_unclipped_brightness_offset_preserves_and_scaling_scales_spread():
    rgb = np.random.default_rng(3).uniform(0.1, 0.3, (256, 256, 3))

    np.testing.assert_allclose(patch_iqr_features(2 * rgb + 0.1), 2 * patch_iqr_features(rgb), atol=1e-12)


def test_physical_features_use_calibrated_center_crop_and_normalized_rgb(tmp_path):
    rgb = np.full((512, 512, 3), 255, dtype=np.uint8)
    rgb[128:256, 128:384] = [0, 51, 204]
    rgb[256:384, 128:384] = [255, 153, 51]
    path = tmp_path / "calibrated.png"
    Image.fromarray(rgb).save(path)
    camera = pd.Series({"ppm": 5.12, "width": 1024, "height": 1024})

    np.testing.assert_allclose(physical_patch_iqr_features(path, camera), [1, 0.4, 0.6, 0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("shape", [(256,), (256, 256), (255, 256, 3), (256, 256, 4)])
def test_wrong_rgb_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="256"):
        patch_iqr_features(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_rgb_is_rejected(value):
    rgb = np.zeros((256, 256, 3))
    rgb[100, 100, 1] = value
    with pytest.raises(ValueError, match="finite"):
        patch_iqr_features(rgb)


@pytest.mark.parametrize("value", [-0.1, 1.1, 255])
def test_rgb_outside_unit_range_is_rejected(value):
    rgb = np.zeros((256, 256, 3))
    rgb[100, 100, 1] = value
    with pytest.raises(ValueError, match=r"\[0, ?1\]"):
        patch_iqr_features(rgb)
