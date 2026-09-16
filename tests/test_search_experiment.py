import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.search_experiment import _submission_queue


def candidate(tmp_path, name, score, camera, value, reverse=False):
    frame = pd.DataFrame(np.tile([value] * 10 + [100.], (2, 1)), columns=CANONICAL_GRAIN_LABELS)
    frame.insert(0, 'sample_id', ['A', 'B'])
    frame.loc[1, list(CANONICAL_GRAIN_LABELS)[:10]] += 1
    if reverse:
        frame = frame.iloc[::-1]
    path = tmp_path / f'{name}.csv'
    frame.to_csv(path, index=False)
    return {'experiment': name, 'loo_emd': score, 'paired_camera_disagreement_emd': camera,
            'submission': {'path': str(path)}}


def test_queue_ranks_then_deduplicates_by_aligned_values(tmp_path):
    old = candidate(tmp_path, 'old', 0, 0, 20)
    duplicate_old = candidate(tmp_path, 'old_copy', 1, 1, 20, reverse=True)
    second = candidate(tmp_path, 'second', 3, 5, 30)
    duplicate_second = candidate(tmp_path, 'duplicate', 4, 1, 30, reverse=True)
    first = candidate(tmp_path, 'first', 3, 4, 40)
    queue, skipped = _submission_queue([duplicate_second, duplicate_old, second, first], [old['submission']['path']])
    assert queue == ['first', 'second']
    assert {row['experiment'] for row in skipped} == {'old_copy', 'duplicate'}


def test_queue_uses_name_for_exact_score_ties_and_absolute_tolerance(tmp_path):
    a = candidate(tmp_path, 'a', 2, 2, 40)
    b = candidate(tmp_path, 'b', 2, 2, 40 + 5e-10)
    c = candidate(tmp_path, 'c', 2, 2, 40 + 1e-7)
    assert _submission_queue([c, b, a], [])[0] == ['a', 'c']


def test_queue_rejects_incompatible_soil_ids(tmp_path):
    a = candidate(tmp_path, 'a', 2, 2, 40)
    b = candidate(tmp_path, 'b', 3, 2, 30)
    path = b['submission']['path']
    frame = pd.read_csv(path)
    frame.loc[0, 'sample_id'] = 'UNKNOWN'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='sample_id'):
        _submission_queue([a, b], [])
