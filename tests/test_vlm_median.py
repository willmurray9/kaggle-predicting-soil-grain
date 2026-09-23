import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS


def frame(ids, values):
    result = pd.DataFrame([[value] * 10 + [100] for value in values],
                          columns=CANONICAL_GRAIN_LABELS)
    result.insert(0, "sample_id", ids)
    return result


def test_median_aligns_soils_and_retains_template_order():
    from soilgrain.vlm_median import aligned_median

    result = aligned_median([
        frame(["a", "b"], [10, 70]), frame(["b", "a"], [90, 40]),
        frame(["a", "b"], [20, 50]),
    ], ["b", "a"])
    assert result.sample_id.tolist() == ["b", "a"]
    np.testing.assert_array_equal(result.iloc[:, 1:].to_numpy(),
                                  [[70] * 10 + [100], [20] * 10 + [100]])


def test_median_rejects_missing_duplicate_or_extra_soils():
    from soilgrain.vlm_median import aligned_median

    correct = frame(["a", "b"], [10, 20])
    for invalid in (frame(["a"], [10]), frame(["a", "a"], [10, 20]),
                    frame(["a", "b", "c"], [10, 20, 30])):
        with pytest.raises(ValueError):
            aligned_median([correct, invalid, correct], ["a", "b"])


def test_median_rejects_invalid_component_even_when_median_would_hide_it():
    from soilgrain.vlm_median import aligned_median

    valid = frame(["a"], [20])
    invalid = frame(["a"], [40])
    invalid.iloc[0, 2] = 30
    with pytest.raises(ValueError):
        aligned_median([valid, invalid, valid], ["a"])
