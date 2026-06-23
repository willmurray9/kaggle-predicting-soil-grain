import pandas as pd
import pytest

from soilgrain.submission import validate_submission


def _sample_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["TEST_01", "TEST_02"],
            "0.002": [0.0, 0.0],
            "0.0063": [0.0, 0.0],
            "0.02": [0.0, 0.0],
            "0.063": [0.0, 0.0],
            "0.2": [0.0, 0.0],
            "0.63": [0.0, 0.0],
            "2": [0.0, 0.0],
            "6.3": [0.0, 0.0],
            "20": [0.0, 0.0],
            "63": [0.0, 0.0],
            "200": [100.0, 100.0],
        }
    )


def test_valid_baseline_submission_passes() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission.loc[:, "0.002":"63"] = 10.0
    submission["200"] = 100.0

    validate_submission(submission, sample)


def test_submission_rejects_missing_sample_ids() -> None:
    sample = _sample_submission()
    submission = sample.iloc[[0]].copy()

    with pytest.raises(ValueError, match="row count"):
        validate_submission(submission, sample)


def test_submission_rejects_non_monotonic_rows() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission.loc[0, "0.002"] = 20.0
    submission.loc[0, "0.0063"] = 10.0

    with pytest.raises(ValueError, match="non-decreasing"):
        validate_submission(submission, sample)


def test_submission_rejects_final_value_not_one_hundred() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission["200"] = 99.9

    with pytest.raises(ValueError, match="final"):
        validate_submission(submission, sample)
