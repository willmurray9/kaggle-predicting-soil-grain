from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import map_coordinates

from soilgrain.spectral_features import SPECTRAL_BANDS_MM


CROP_MM = 100
CROP_PIXELS = 460
ABSOLUTE_LAGS_PIXELS = tuple(round(lag * CROP_PIXELS / 256) for lag in (1, 4, 16))
TEXTURE_OFFSETS_MM = (1., 2., 4., 8.)
TEXTURE_LAGS_PIXELS = tuple(round(offset * CROP_PIXELS / CROP_MM) for offset in TEXTURE_OFFSETS_MM)
LBP_RADII_MM = (0.5, 1., 2., 4.)


def native_crop(path: str | Path, camera: pd.Series) -> Image.Image:
    """Calibrated 100 mm center crop, EXIF oriented and rendered at 460 pixels."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    calibration = np.asarray([camera["ppm"], camera["width"], camera["height"]], dtype=float)
    if not np.all(np.isfinite(calibration) & (calibration > 0)):
        raise ValueError("Native crop requires a positive finite camera scale and dimensions")
    ppm = calibration[0] * max(image.size) / max(calibration[1:])
    side = round(CROP_MM * ppm)
    if side < 1 or side > min(image.size):
        raise ValueError(f"Cannot extract a 100 mm crop from {path} with PPM={ppm}")
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (CROP_PIXELS, CROP_PIXELS), Image.Resampling.LANCZOS,
    )


def _gray_image(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=float)
    if gray.shape != (CROP_PIXELS, CROP_PIXELS) or not np.isfinite(gray).all():
        raise ValueError("Native texture requires a finite 460 by 460 grayscale image")
    return gray


def native_spectrum_features(gray: np.ndarray) -> np.ndarray:
    """Six normalized wavelength powers using the 100/460 mm pixel spacing."""
    gray = _gray_image(gray)
    window = np.outer(np.hanning(CROP_PIXELS), np.hanning(CROP_PIXELS))
    centered = gray - (gray * window).sum() / window.sum()
    power = np.abs(np.fft.fft2(centered * window)) ** 2
    frequency = np.fft.fftfreq(CROP_PIXELS, d=CROP_MM / CROP_PIXELS)
    radius = np.hypot(frequency[:, None], frequency[None, :])
    bands = np.array([power[(radius >= 1 / high) & (radius < 1 / low)].sum()
                      for low, high in SPECTRAL_BANDS_MM])
    total = bands.sum()
    return bands / total if total > 1e-12 else np.zeros(len(SPECTRAL_BANDS_MM))


def native_spectral_features(path: str | Path, camera: pd.Series) -> np.ndarray:
    """RGB13, four normalized physical differences and six powers at 460 pixels."""
    rgb = np.asarray(native_crop(path, camera), dtype=float) / 255.
    gray = rgb.mean(axis=2)
    color = np.quantile(rgb, [0.1, 0.5, 0.9], axis=(0, 1)).ravel()
    absolute = np.array([(np.abs(gray[lag:] - gray[:-lag]).mean()
                         + np.abs(gray[:, lag:] - gray[:, :-lag]).mean()) / 2
                        for lag in ABSOLUTE_LAGS_PIXELS])
    variance = gray.var()
    normalized = (np.array([(np.square(gray[lag:] - gray[:-lag]).mean()
                             + np.square(gray[:, lag:] - gray[:, :-lag]).mean()) / (4 * variance)
                            for lag in TEXTURE_LAGS_PIXELS])
                  if variance >= 1e-12 else np.zeros(len(TEXTURE_LAGS_PIXELS)))
    return np.concatenate([color, [gray.std()], absolute, normalized, native_spectrum_features(gray)])


def lbp_features(gray: np.ndarray) -> np.ndarray:
    """Four rotation-invariant uniform 8-neighbor histograms at physical radii."""
    gray = _gray_image(gray)
    # Affine normalization preserves ordering; tolerance treats interpolation roundoff as equality.
    gray = gray - gray.min()
    span = gray.max()
    if span > 0:
        gray = gray / span
    histograms = []
    for radius_mm in LBP_RADII_MM:
        radius = radius_mm * CROP_PIXELS / CROP_MM
        margin = int(np.ceil(radius))
        y, x = np.mgrid[margin:CROP_PIXELS - margin, margin:CROP_PIXELS - margin]
        center = gray[margin:-margin, margin:-margin]
        bits = np.array([
            map_coordinates(gray, [y + radius * np.sin(angle), x + radius * np.cos(angle)],
                            order=1, mode="nearest", prefilter=False) >= center - 1e-12
            for angle in np.arange(8) * np.pi / 4
        ])
        transitions = np.count_nonzero(bits != np.roll(bits, 1, axis=0), axis=0)
        labels = np.where(transitions <= 2, bits.sum(axis=0), 9)
        histogram = np.bincount(labels.ravel(), minlength=10).astype(float)
        histograms.append(histogram / histogram.sum())
    return np.concatenate(histograms)


def physical_lbp_features(path: str | Path, camera: pd.Series) -> np.ndarray:
    """Forty brightness-order features from the calibrated 460-pixel crop."""
    gray = (np.asarray(native_crop(path, camera), dtype=float) / 255.).mean(axis=2)
    return lbp_features(gray)
