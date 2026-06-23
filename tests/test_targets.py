import pandas as pd
import pytest

from soilgrain.targets import cumulative_to_bin_masses, ordered_grain_columns, validate_cumulative_curves


def test_cumulative_to_bin_masses_preserves_total_mass() -> None:
    curves = pd.DataFrame(
        {
            "sample_id": ["A"],
            "0.002": [2.0],
            "0.0063": [5.0],
            "0.02": [11.0],
            "0.063": [20.0],
            "0.2": [35.0],
            "0.63": [50.0],
            "2": [70.0],
            "6.3": [85.0],
            "20": [95.0],
            "63": [99.0],
            "200": [100.0],
        }
    )

    masses = cumulative_to_bin_masses(curves)

    assert masses.shape == (1, 11)
    assert masses.iloc[0].sum() == 100.0
    assert masses.iloc[0, 0] == 2.0
    assert masses.iloc[0, 1] == 3.0


def test_validate_cumulative_curves_rejects_decreasing_rows() -> None:
    curves = pd.DataFrame(
        {
            "sample_id": ["A"],
            "0.002": [0.0],
            "0.0063": [8.0],
            "0.02": [7.0],
            "0.063": [20.0],
            "0.2": [35.0],
            "0.63": [50.0],
            "2": [70.0],
            "6.3": [85.0],
            "20": [95.0],
            "63": [99.0],
            "200": [100.0],
        }
    )

    with pytest.raises(ValueError, match="non-decreasing"):
        validate_cumulative_curves(curves)


def test_grain_headers_normalize_numeric_values_without_losing_output_names() -> None:
    sample = pd.DataFrame(columns=["sample_id", "0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"])

    assert ordered_grain_columns(sample) == ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
