from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from soilgrain.constants import SUPPORT_DIAMETERS


def _as_float(label: object) -> float | None:
    try:
        return float(str(label))
    except ValueError:
        return None


def _support_key(value: float) -> float | None:
    for support in SUPPORT_DIAMETERS:
        if np.isclose(value, support, rtol=0.0, atol=max(1e-12, support * 1e-9)):
            return support
    return None


def ordered_grain_columns(df: pd.DataFrame | Sequence[object]) -> list[str]:
    columns = list(df.columns if isinstance(df, pd.DataFrame) else df)
    by_support: dict[float, str] = {}
    for col in columns:
        value = _as_float(col)
        if value is None:
            continue
        support = _support_key(value)
        if support is not None:
            by_support[support] = str(col)

    missing = [str(s) for s in SUPPORT_DIAMETERS if s not in by_support]
    if missing:
        raise ValueError(f"Missing grain-size columns for support diameters: {missing}")
    return [by_support[s] for s in SUPPORT_DIAMETERS]


def curve_array(df: pd.DataFrame) -> np.ndarray:
    return df[ordered_grain_columns(df)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def validate_cumulative_curves(df: pd.DataFrame, *, require_final_100: bool = True) -> None:
    values = curve_array(df)
    if np.isnan(values).any():
        raise ValueError("Cumulative curves contain null or non-numeric values.")
    if ((values < 0) | (values > 100)).any():
        raise ValueError("Cumulative curves must stay within [0, 100].")
    if (np.diff(values, axis=1) < -1e-9).any():
        raise ValueError("Cumulative curves must be non-decreasing.")
    if require_final_100 and not np.allclose(values[:, -1], 100.0, atol=1e-9):
        raise ValueError("Cumulative curves must have final 200 mm value equal to 100.")


def cumulative_to_bin_masses(df: pd.DataFrame) -> pd.DataFrame:
    validate_cumulative_curves(df)
    cumulative = curve_array(df)
    masses = np.column_stack([cumulative[:, 0], np.diff(cumulative, axis=1)])
    labels = [
        "<=0.002",
        "0.002-0.0063",
        "0.0063-0.02",
        "0.02-0.063",
        "0.063-0.2",
        "0.2-0.63",
        "0.63-2",
        "2-6.3",
        "6.3-20",
        "20-63",
        "63-200",
    ]
    return pd.DataFrame(masses, columns=labels, index=df.index)
