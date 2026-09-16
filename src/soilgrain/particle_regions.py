from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage


def physical_source_crop(path: str | Path, camera: pd.Series) -> tuple[np.ndarray, float]:
    """Central calibrated 100 mm RGB crop, preserving available source pixels."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    calibration = np.array([camera["ppm"], camera["width"], camera["height"]], dtype=float)
    if not np.isfinite(calibration).all() or np.any(calibration <= 0):
        raise ValueError("Camera PPM and native dimensions must be finite and positive.")
    ppm = float(calibration[0] * max(image.size) / max(calibration[1:]))
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("Effective PPM must be finite and positive.")
    side = round(100 * ppm)
    if side < 1 or side > min(image.size):
        raise ValueError(f"Cannot extract a 100 mm crop from {path} with PPM={ppm}.")
    left, top = (image.width - side) // 2, (image.height - side) // 2
    crop = image.crop((left, top, left + side, top + side))
    return np.asarray(crop, dtype=float) / 255.0, ppm


def particle_regions(rgb: np.ndarray, ppm: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed gradient-watershed proposals; regions are not necessarily grains."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected a nonempty height by width RGB image.")
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("RGB values must be finite and within [0, 1].")
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("PPM must be finite and positive.")

    empty = np.zeros(rgb.shape[:2], dtype=np.int32)
    gradient = ndimage.gaussian_gradient_magnitude(
        rgb.mean(axis=2), sigma=max(0.5, 0.25 * ppm), mode="reflect",
    )
    maximum = gradient.max()
    if maximum <= 1e-12:
        return empty, empty.copy()
    radius = 0.5 * ppm
    coordinates = np.arange(-int(np.ceil(radius)), int(np.ceil(radius)) + 1)
    disk = coordinates[:, None] ** 2 + coordinates[None, :] ** 2 <= radius ** 2
    interiors = gradient <= np.quantile(gradient, 0.25, method="linear")
    opened = ndimage.binary_opening(interiors, structure=disk, iterations=1, border_value=0)
    connectivity = np.ones((3, 3), dtype=bool)
    seeds, count = ndimage.label(opened, structure=connectivity)
    seeds = seeds.astype(np.int32, copy=False)
    if count == 0:
        return empty, seeds
    elevation = np.rint(gradient / maximum * 65535).astype(np.uint16)
    labels = ndimage.watershed_ift(elevation, seeds, structure=connectivity)
    return labels, seeds


def region_measurements(labels: np.ndarray, ppm: float) -> pd.DataFrame:
    """Calibrated projected areas and equivalent-circle diameters for every region."""
    labels = np.asarray(labels)
    if (labels.ndim != 2 or min(labels.shape) < 1
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError("Region labels must be a nonempty 2D array of nonnegative integers.")
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError("PPM must be finite and positive.")
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    edge_ids = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    touches_border = np.isin(ids, edge_ids)
    area_mm2 = counts / ppm ** 2
    diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
    return pd.DataFrame({
        "region_id": ids, "area_pixels": counts, "area_mm2": area_mm2,
        "diameter_mm": diameter_mm, "touches_border": touches_border,
        "coarse_interior": (diameter_mm >= 2) & ~touches_border,
    })
