from pathlib import Path

import numpy as np
import pandas as pd

from soilgrain.image_model import physical_crop


TEXTURE_OFFSETS_MM = (1.0, 2.0, 4.0, 8.0)
TEXTURE_LAGS_PIXELS = tuple(round(offset * 256 / 100) for offset in TEXTURE_OFFSETS_MM)


def physical_texture_features(path: str | Path, camera: pd.Series) -> np.ndarray:
    """Variance-normalized squared differences at four physical scales in a 100 mm crop."""
    crop = physical_crop(path, camera, crop_mm=100)
    gray = (np.asarray(crop, dtype=float) / 255.0).mean(axis=2)
    variance = gray.var()
    if variance < 1e-12:
        return np.zeros(len(TEXTURE_LAGS_PIXELS))
    return np.array([
        (
            np.square(gray[lag:] - gray[:-lag]).mean()
            + np.square(gray[:, lag:] - gray[:, :-lag]).mean()
        ) / (4 * variance)
        for lag in TEXTURE_LAGS_PIXELS
    ])
