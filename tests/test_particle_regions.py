import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain.particle_regions import particle_regions, physical_source_crop, region_measurements


def test_source_crop_corrects_resolution_and_preserves_pixels_without_resizing(tmp_path):
    pixels = np.random.default_rng(9).integers(0, 256, (401, 501, 3), dtype=np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    camera = pd.Series({"ppm": 5, "width": 1002, "height": 802})

    rgb, ppm = physical_source_crop(path, camera)

    assert ppm == 2.5
    assert rgb.shape == (250, 250, 3) and rgb.dtype == np.float64
    np.testing.assert_array_equal(rgb, pixels[75:325, 125:375] / 255)


def test_source_crop_applies_exif_orientation_before_cropping(tmp_path):
    oriented = np.random.default_rng(2).integers(0, 256, (600, 400, 3), dtype=np.uint8)
    image = Image.fromarray(np.rot90(oriented))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    camera = pd.Series({"ppm": 4, "width": 1200, "height": 800})

    rgb, ppm = physical_source_crop(path, camera)

    assert ppm == 2
    np.testing.assert_array_equal(rgb, oriented[200:400, 100:300] / 255)


@pytest.mark.parametrize("ppm", [0, -1, np.nan, np.inf, 3])
def test_source_crop_rejects_invalid_scale_or_field_that_does_not_fit(tmp_path, ppm):
    path = tmp_path / "source.png"
    Image.new("RGB", (200, 200)).save(path)
    camera = pd.Series({"ppm": ppm, "width": 200, "height": 200})
    with pytest.raises(ValueError):
        physical_source_crop(path, camera)


@pytest.mark.parametrize("level", [0, 0.4, 1])
def test_flat_images_have_no_regions_or_seeds(level):
    labels, seeds = particle_regions(np.full((20, 30, 3), level), 4.6)

    for result in (labels, seeds):
        assert result.shape == (20, 30) and result.dtype == np.int32
        np.testing.assert_array_equal(result, 0)


def test_negligible_gradient_is_not_amplified_into_regions():
    rgb = np.zeros((20, 20, 3))
    rgb[10, 10] = 1e-13

    labels, seeds = particle_regions(rgb, 1)

    np.testing.assert_array_equal(labels, 0)
    np.testing.assert_array_equal(seeds, 0)


def test_diagonal_low_gradient_interiors_form_one_eight_connected_seed():
    rgb = np.ones((3, 3, 3))
    rgb[1, 1] = 0

    labels, seeds = particle_regions(rgb, 1)

    np.testing.assert_array_equal(seeds, [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    np.testing.assert_array_equal(labels, np.ones((3, 3)))
    assert labels.dtype == seeds.dtype == np.int32


def test_physical_opening_removes_isolated_seeds():
    rgb = np.ones((3, 3, 3))
    rgb[1, 1] = 0

    labels, seeds = particle_regions(rgb, 2)

    np.testing.assert_array_equal(labels, 0)
    np.testing.assert_array_equal(seeds, 0)


def test_radius_one_disk_opening_cuts_seed_corners_at_zero_image_border():
    rgb = np.zeros((20, 20, 3))
    rgb[:, 10:] = 1

    _, seeds = particle_regions(rgb, 2)

    assert seeds[0, 0] == 0
    assert seeds[0, 1] > 0 and seeds[1, 0] > 0
    assert seeds[0, 1] == seeds[1, 0]


def test_separated_discs_retain_distinct_seed_ids_and_partition_the_image():
    y, x = np.indices((80, 100))
    gray = np.zeros((80, 100))
    gray[(x - 25) ** 2 + (y - 40) ** 2 <= 100] = 0.6
    gray[(x - 75) ** 2 + (y - 40) ** 2 <= 100] = 1
    rgb = np.repeat(gray[..., None], 3, axis=2)

    labels, seeds = particle_regions(rgb, 2)

    assert seeds[40, 25] > 0 and seeds[40, 75] > 0
    assert seeds[40, 25] != seeds[40, 75]
    assert np.all(labels > 0)
    np.testing.assert_array_equal(labels[seeds > 0], seeds[seeds > 0])
    np.testing.assert_array_equal(np.unique(labels), np.unique(seeds[seeds > 0]))
    again = particle_regions(rgb, 2)
    np.testing.assert_array_equal(again[0], labels)
    np.testing.assert_array_equal(again[1], seeds)


def test_regions_use_rgb_mean_not_channel_order():
    rgb = np.random.default_rng(3).uniform(0, 1, (25, 35, 3))

    original = particle_regions(rgb, 1)
    reversed_channels = particle_regions(rgb[..., ::-1], 1)

    for expected, actual in zip(original, reversed_channels, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_region_table_calibrates_sparse_ids_and_preserves_small_or_border_regions():
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[0, :4] = 9
    labels[2:4, 2:4] = 2
    labels[2, 5] = 42

    table = region_measurements(labels, 1)

    assert list(table.columns) == [
        "region_id", "area_pixels", "area_mm2", "diameter_mm", "touches_border", "coarse_interior",
    ]
    assert table.region_id.tolist() == [2, 9, 42]
    assert table.area_pixels.tolist() == [4, 4, 1]
    np.testing.assert_allclose(table.area_mm2, [4, 4, 1])
    np.testing.assert_allclose(table.diameter_mm, [4 / np.sqrt(np.pi), 4 / np.sqrt(np.pi), 2 / np.sqrt(np.pi)])
    assert table.touches_border.tolist() == [False, True, False]
    assert table.coarse_interior.tolist() == [True, False, False]
    finer = region_measurements(labels, 2)
    np.testing.assert_allclose(finer.area_mm2, table.area_mm2 / 4)
    np.testing.assert_allclose(finer.diameter_mm, table.diameter_mm / 2)
    assert not finer.coarse_interior.any()


def test_empty_region_table_retains_its_schema():
    table = region_measurements(np.zeros((10, 10), dtype=np.int32), 1)

    assert table.empty
    assert list(table.columns) == [
        "region_id", "area_pixels", "area_mm2", "diameter_mm", "touches_border", "coarse_interior",
    ]


@pytest.mark.parametrize("ppm", [0, -1, np.nan, np.inf])
def test_detector_and_measurements_reject_invalid_ppm(ppm):
    with pytest.raises(ValueError, match="PPM"):
        particle_regions(np.zeros((4, 4, 3)), ppm)
    with pytest.raises(ValueError, match="PPM"):
        region_measurements(np.zeros((4, 4), dtype=np.int32), ppm)


@pytest.mark.parametrize("shape", [(4,), (4, 4), (4, 4, 4), (0, 4, 3)])
def test_detector_rejects_invalid_rgb_shape(shape):
    with pytest.raises(ValueError, match="RGB"):
        particle_regions(np.zeros(shape), 1)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_detector_rejects_nonfinite_or_unnormalized_rgb(value):
    rgb = np.zeros((4, 4, 3))
    rgb[1, 1, 0] = value
    with pytest.raises(ValueError):
        particle_regions(rgb, 1)


@pytest.mark.parametrize("labels", [np.zeros(3, dtype=int), np.zeros((0, 3), dtype=int),
                                   np.array([[0, -1]]), np.array([[0.0, 1.5]]), np.array([[np.nan]])])
def test_region_measurements_reject_invalid_labels(labels):
    with pytest.raises(ValueError, match="labels"):
        region_measurements(labels, 1)
