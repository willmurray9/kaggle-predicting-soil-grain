from __future__ import annotations

import numpy as np
import pandas as pd

from soilgrain.targets import ordered_grain_columns, validate_cumulative_curves


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    expected_columns = list(sample_submission.columns)
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must exactly match sample submission: {expected_columns}")
    if "sample_id" not in submission.columns:
        raise ValueError("Submission must contain sample_id.")
    if sample_submission["sample_id"].duplicated().any():
        raise ValueError("Sample submission contains duplicate sample_id values.")
    if submission["sample_id"].duplicated().any():
        raise ValueError("Submission contains duplicate sample_id values.")
    if len(submission) != len(sample_submission):
        raise ValueError(f"Submission row count mismatch: got={len(submission)}, expected={len(sample_submission)}")

    expected_ids = set(sample_submission["sample_id"].astype(str))
    got_ids = set(submission["sample_id"].astype(str))
    if expected_ids != got_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        raise ValueError(f"Submission sample_id mismatch: missing={missing[:5]}, extra={extra[:5]}")

    grain_cols = ordered_grain_columns(sample_submission)
    numeric = submission[grain_cols].apply(pd.to_numeric, errors="coerce")
    checked = submission.copy()
    checked[grain_cols] = numeric
    validate_cumulative_curves(checked)
    if not np.allclose(numeric[grain_cols[-1]], 100.0, atol=1e-9):
        raise ValueError("Submission final 200 mm column must equal 100.")


def submission_from_curve(sample_submission: pd.DataFrame, curve: np.ndarray) -> pd.DataFrame:
    grain_cols = ordered_grain_columns(sample_submission)
    values = np.asarray(curve, dtype=float)
    if values.shape != (len(grain_cols),):
        raise ValueError(f"Expected curve with {len(grain_cols)} values, got shape {values.shape}")

    out = sample_submission.copy()
    for col, value in zip(grain_cols, values, strict=True):
        out[col] = float(value)
    out[grain_cols[-1]] = 100.0
    validate_submission(out, sample_submission)
    return out
