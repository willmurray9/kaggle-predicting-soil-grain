import numpy as np

from soilgrain.constants import SUPPORT_DIAMETERS
from soilgrain.metrics import emd_score


def test_emd_returns_zero_for_perfect_prediction() -> None:
    y = np.array([[0, 5, 10, 20, 35, 50, 70, 85, 95, 99, 100]], dtype=float)

    assert emd_score(y, y) == 0.0


def test_emd_all_zero_vs_all_hundred_returns_metric_max() -> None:
    zeros = np.zeros((1, len(SUPPORT_DIAMETERS)))
    hundreds = np.full((1, len(SUPPORT_DIAMETERS)), 100.0)

    assert emd_score(zeros, hundreds) == 500.0


def test_emd_matches_manual_log_width_calculation() -> None:
    true = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]], dtype=float)
    pred = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100]], dtype=float)
    widths = np.diff(np.log10(np.array(SUPPORT_DIAMETERS)))
    expected = float(np.sum(np.abs(true[:, :-1] - pred[:, :-1]) * widths))

    assert emd_score(true, pred) == expected
