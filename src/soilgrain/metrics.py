from __future__ import annotations

import numpy as np

from soilgrain.constants import SUPPORT_DIAMETERS


def emd_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    diameters: tuple[float, ...] = SUPPORT_DIAMETERS,
) -> float:
    """Mean log-weighted Earth Mover's Distance for cumulative PSD curves."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: true={true.shape}, pred={pred.shape}")
    if true.ndim == 1:
        true = true.reshape(1, -1)
        pred = pred.reshape(1, -1)
    if true.shape[1] != len(diameters):
        raise ValueError(f"Expected {len(diameters)} support points, got {true.shape[1]}")

    widths = np.diff(np.log10(np.asarray(diameters, dtype=float)))
    scores = np.sum(np.abs(true[:, :-1] - pred[:, :-1]) * widths, axis=1)
    return float(np.mean(scores))
