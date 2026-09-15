from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import grey_opening

from soilgrain.image_model import physical_crop


GRANULOMETRY_SIZES_PIXELS = (3, 5, 11, 21, 41, 83)


def granulometry_features(gray: np.ndarray) -> np.ndarray:
    """Twelve normalized bright/dark opening losses from a 100 mm, 256-pixel crop."""
    gray = np.asarray(gray, dtype=float)
    if gray.shape != (256, 256) or not np.isfinite(gray).all():
        raise ValueError("Granulometry requires a finite 256 by 256 grayscale image.")
    padded = np.pad(gray, 83, mode="reflect")
    patterns = []
    for image in (padded, 1 - padded):
        volumes = [image[83:-83, 83:-83].sum()]
        volumes.extend(
            grey_opening(image, size=(size, size))[83:-83, 83:-83].sum()
            for size in GRANULOMETRY_SIZES_PIXELS
        )
        losses = np.maximum(-np.diff(volumes), 0.0)
        total = losses.sum()
        patterns.append(losses / total if total > 1e-12 else np.zeros(6))
    return np.concatenate(patterns)


def physical_granulometry_features(path: str | Path, camera: pd.Series) -> np.ndarray:
    """Calibrated shape probes measure image contrast, not bulk grain mass."""
    crop = physical_crop(path, camera, crop_mm=100)
    gray = (np.asarray(crop, dtype=float) / 255.0).mean(axis=2)
    return granulometry_features(gray)
