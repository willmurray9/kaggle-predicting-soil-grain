import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS


def frame(ids, values):
    result = pd.DataFrame([[v]*10+[100.] for v in values], columns=CANONICAL_GRAIN_LABELS)
    result.insert(0, 'sample_id', ids)
    return result


def test_blend_aligns_soils_and_scores_the_actual_averaged_curves():
    from soilgrain.vlm_mass_blend import blend_frames

    truth = frame(['S0', 'S1'], [10, 70])
    sample = frame(['T2', 'T1'], [0, 0])
    vlm = {'oof': frame(['S1', 'S0'], [80, 20]), 'submission': frame(['T1', 'T2'], [20, 60])}
    mass = {'oof': frame(['S0', 'S1'], [40, 20]), 'submission': frame(['T2', 'T1'], [40, 0])}
    oof, submission = blend_frames(truth, sample, vlm, mass)
    assert oof.sample_id.tolist() == ['S0', 'S1']
    np.testing.assert_allclose(oof[list(CANONICAL_GRAIN_LABELS[:10])],
                               np.repeat([[30.], [50.]], 10, axis=1))
    np.testing.assert_allclose(oof.emd, [100., 100.])
    np.testing.assert_allclose(oof.vlm_emd, [50., 50.])
    np.testing.assert_allclose(oof.mass_emd, [150., 250.])
    assert submission.sample_id.tolist() == ['T2', 'T1']
    np.testing.assert_allclose(submission.iloc[:, 1:11], np.repeat([[50.], [10.]], 10, axis=1))


def test_blend_rejects_missing_heldout_soil_even_when_components_agree():
    from soilgrain.vlm_mass_blend import blend_frames

    truth = frame(['S0', 'S1'], [10, 70])
    sample = frame(['T'], [0])
    incomplete = {'oof': frame(['S0'], [20]), 'submission': frame(['T'], [20])}
    with pytest.raises(ValueError):
        blend_frames(truth, sample, incomplete, incomplete)
