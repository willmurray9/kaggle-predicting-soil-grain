import numpy as np
import pandas as pd
import pytest
from PIL import Image


def test_native_crop_scales_calibration_and_preserves_center_pixels(tmp_path):
    from soilgrain.native_texture import native_crop
    pixels = np.random.default_rng(1).integers(0, 256, (601, 801, 3), dtype=np.uint8)
    path = tmp_path / "resized.png"
    Image.fromarray(pixels).save(path)
    camera = pd.Series({"ppm": 10., "width": 1602, "height": 1202})
    expected = Image.fromarray(pixels[50:550, 150:650]).resize((460, 460), Image.Resampling.LANCZOS)
    np.testing.assert_array_equal(np.asarray(native_crop(path, camera)), np.asarray(expected))


def test_native_crop_follows_exif_orientation(tmp_path):
    from soilgrain.native_texture import native_crop
    oriented = np.random.default_rng(2).integers(0, 256, (800, 600, 3), dtype=np.uint8)
    image = Image.fromarray(np.rot90(oriented))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    camera = pd.Series({"ppm": 9.2, "width": 1600, "height": 1200})
    np.testing.assert_array_equal(np.asarray(native_crop(path, camera)), oriented[170:630, 70:530])


@pytest.mark.parametrize("ppm", [0., -1., np.nan, np.inf, 100.])
def test_native_crop_rejects_invalid_or_unavailable_physical_scale(tmp_path, ppm):
    from soilgrain.native_texture import native_crop
    path = tmp_path / "small.png"
    Image.new("RGB", (460, 460)).save(path)
    camera = pd.Series({"ppm": ppm, "width": 460, "height": 460})
    with pytest.raises(ValueError, match="scale|crop"):
        native_crop(path, camera)


@pytest.mark.parametrize("level", [0., 0.2, 1., 255., -3.])
def test_flat_lbp_has_eight_equal_neighbors_at_every_radius(level):
    from soilgrain.native_texture import lbp_features
    result = lbp_features(np.full((460, 460), level)).reshape(4, 10)
    expected = np.zeros((4, 10))
    expected[:, 8] = 1
    np.testing.assert_array_equal(result, expected)


def test_lbp_linear_ramp_has_four_uniform_neighbors_at_every_radius():
    from soilgrain.native_texture import lbp_features
    y, x = np.indices((460, 460))
    result = lbp_features(x + 0.3 * y).reshape(4, 10)
    expected = np.zeros((4, 10))
    expected[:, 4] = 1
    np.testing.assert_array_equal(result, expected)


def test_lbp_step_respects_each_physical_radius_and_excludes_boundary_centers():
    from soilgrain.native_texture import lbp_features
    gray = np.zeros((460, 460))
    gray[:, 230:] = 1
    result = lbp_features(gray).reshape(4, 10)
    expected = np.zeros((4, 10))
    expected[:, 5] = [2 / 454, 4 / 450, 7 / 440, 14 / 422]
    expected[:, 7] = [1 / 454, 1 / 450, 3 / 440, 5 / 422]
    expected[:, 8] = [451 / 454, 445 / 450, 430 / 440, 403 / 422]
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_lbp_is_invariant_to_positive_affine_gray_and_quarter_turn():
    from soilgrain.native_texture import lbp_features
    gray = np.random.default_rng(3).uniform(0.1, 0.3, (460, 460))
    original = lbp_features(gray)
    np.testing.assert_array_equal(original, lbp_features(3.7 * gray + 9))
    np.testing.assert_allclose(original, lbp_features(np.rot90(gray)), atol=1e-12)
    assert original.shape == (40,)
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original.reshape(4, 10).sum(axis=1), 1)
    assert np.all(original.reshape(4, 10)[:, 9] > 0)  # Nonuniform patterns retain their own bin.


@pytest.mark.parametrize("band,cycles", list(enumerate([75, 36, 18, 9, 4, 2])))
def test_native_spectrum_uses_physical_frequency_bands(band, cycles):
    from soilgrain.native_texture import native_spectrum_features
    gray = np.tile(0.5 + 0.2 * np.sin(2 * np.pi * cycles * np.arange(460) / 460), (460, 1))
    features = native_spectrum_features(gray)
    assert features.shape == (6,)
    assert np.isfinite(features).all()
    assert np.argmax(features) == band
    assert features[band] > 0.8
    assert features.sum() == pytest.approx(1)


def test_native_spectral23_retains_physical_lags_on_known_step_image(tmp_path):
    from soilgrain.native_texture import native_spectral_features
    gray = np.zeros((460, 460), dtype=np.uint8)
    gray[:, 230:] = 255
    path = tmp_path / "step.png"
    Image.fromarray(gray).save(path)
    camera = pd.Series({"ppm": 4.6, "width": 460, "height": 460})
    features = native_spectral_features(path, camera)
    assert features.shape == (23,)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:10], [0, 0, 0, .5, .5, .5, 1, 1, 1, .5])
    np.testing.assert_allclose(features[10:13], [2 / (2 * 458), 7 / (2 * 453), 29 / (2 * 431)])
    np.testing.assert_allclose(features[13:17], [5 / 455, 9 / 451, 18 / 442, 37 / 423])
    assert features[17:].sum() == pytest.approx(1)


def test_public_extractors_share_calibrated_rgb_mean_and_are_finite_for_flat_images(tmp_path):
    from soilgrain.native_texture import native_spectral_features, physical_lbp_features
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[70:530, 170:630] = [20, 80, 140]
    path = tmp_path / "flat_center.png"
    Image.fromarray(image).save(path)
    camera = pd.Series({"ppm": 9.2, "width": 1600, "height": 1200})
    spectral, lbp = native_spectral_features(path, camera), physical_lbp_features(path, camera)
    np.testing.assert_allclose(spectral[:9], np.tile([20, 80, 140], 3) / 255)
    np.testing.assert_allclose(spectral[9:], 0, atol=1e-12)
    expected = np.zeros((4, 10))
    expected[:, 8] = 1
    np.testing.assert_array_equal(lbp.reshape(4, 10), expected)


@pytest.mark.parametrize("shape", [(256, 256), (460,), (460, 460, 3)])
def test_array_extractors_reject_wrong_shape(shape):
    from soilgrain.native_texture import lbp_features, native_spectrum_features
    for extract in (lbp_features, native_spectrum_features):
        with pytest.raises(ValueError, match="460"):
            extract(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_array_extractors_reject_nonfinite_pixels(value):
    from soilgrain.native_texture import lbp_features, native_spectrum_features
    gray = np.zeros((460, 460))
    gray[200, 200] = value
    for extract in (lbp_features, native_spectrum_features):
        with pytest.raises(ValueError, match="finite"):
            extract(gray)
