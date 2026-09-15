from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.image_model import physical_crop


SPECTRAL_BANDS_MM = ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 64.0))


def spectrum_features(gray: np.ndarray) -> np.ndarray:
    """Six normalized image-wavelength powers from a 100 mm, 256-pixel crop."""
    gray = np.asarray(gray, dtype=float)
    if gray.shape != (256, 256) or not np.isfinite(gray).all():
        raise ValueError("Spectrum requires a finite 256 by 256 grayscale image.")
    window = np.outer(np.hanning(256), np.hanning(256))
    centered = gray - (gray * window).sum() / window.sum()
    power = np.abs(np.fft.fft2(centered * window)) ** 2
    frequency = np.fft.fftfreq(256, d=100 / 256)
    radius = np.hypot(frequency[:, None], frequency[None, :])
    bands = np.array([
        power[(radius >= 1 / high) & (radius < 1 / low)].sum()
        for low, high in SPECTRAL_BANDS_MM
    ])
    total = bands.sum()
    return bands / total if total > 1e-12 else np.zeros(len(SPECTRAL_BANDS_MM))


def physical_spectrum_features(path: str | Path, camera: pd.Series) -> np.ndarray:
    """Calibrated spectral texture; wavelengths are not particle diameters."""
    crop = physical_crop(path, camera, crop_mm=100)
    gray = (np.asarray(crop, dtype=float) / 255.0).mean(axis=2)
    return spectrum_features(gray)
