from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.image_model import physical_crop


def patch_iqr_features(rgb: np.ndarray) -> np.ndarray:
    """Six color/texture IQRs across sixteen fixed tiles of a 100 mm crop."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.shape != (256, 256, 3) or not np.isfinite(rgb).all():
        raise ValueError("Patch features require a finite 256 by 256 RGB image.")
    if np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("Patch RGB values must lie within [0, 1].")

    descriptors = []
    for row in range(0, 256, 64):
        for column in range(0, 256, 64):
            tile = rgb[row:row + 64, column:column + 64]
            gray = tile.mean(axis=2)
            texture = [
                (np.abs(gray[lag:] - gray[:-lag]).mean()
                 + np.abs(gray[:, lag:] - gray[:, :-lag]).mean()) / 2
                for lag in (1, 4, 16)
            ]
            descriptors.append(np.r_[tile.mean(axis=(0, 1)), texture])
    lower, upper = np.quantile(descriptors, [0.25, 0.75], axis=0, method="linear")
    return upper - lower


def physical_patch_iqr_features(path: str | Path, camera: pd.Series) -> np.ndarray:
    """Regional variation from the existing calibrated 100 mm RGB crop."""
    crop = physical_crop(path, camera, crop_mm=100)
    return patch_iqr_features(np.asarray(crop, dtype=float) / 255.0)
