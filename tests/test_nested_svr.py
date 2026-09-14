import numpy as np
import pandas as pd
import pytest

import soilgrain.nested_svr as nested
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.svr import svr_curves


@pytest.fixture
def soils():
    sample_ids = ["D", "B", "E", "A", "C"]
    features = np.array([[0, 2], [1, 3], [3, 1], [4, 7], [8, 4]], dtype=float)
    curves = np.array([list(np.linspace(v, 90, 10)) + [100] for v in [2, 30, 12, 55, 65]])
    photos = pd.DataFrame([
        {"sample_id": sample_id, "camera": camera,
         "feature_0": values[0] + offset, "feature_1": values[1] - offset}
        for sample_id, values in zip(sample_ids, features, strict=True)
        for camera, offset in [("phone_a", -0.25), ("phone_b", 0.25)]
    ])
    return photos, sample_ids, features, curves


def test_inner_selection_refits_only_inner_training_rows(monkeypatch, soils):
    _, _, features, curves = soils
    calls = []

    def observe(x, y, query, *, C):
        calls.append((x.copy(), y.copy(), query.copy(), C))
        return svr_curves(x, y, query, C=C)

    monkeypatch.setattr(nested, "svr_curves", observe)
    chosen, scores = nested.select_svr_c(features, curves)

    assert len(calls) == len(features) * 3
    expected = {}
    for C in (0.1, 1.0, 10.0):
        setting_calls = [call for call in calls if call[3] == C]
        assert len(setting_calls) == len(features)
        predictions = []
        for i, (x, y, query, _) in enumerate(setting_calls):
            keep = np.arange(len(features)) != i
            np.testing.assert_array_equal(x, features[keep])
            np.testing.assert_array_equal(y, curves[keep])
            np.testing.assert_array_equal(query, features[i:i + 1])
            predictions.append(svr_curves(x, y, query, C=C)[0])
        expected[C] = emd_score(curves, np.vstack(predictions))
    assert scores == expected
    assert chosen == min(expected, key=lambda C: (expected[C], C))


def test_outer_camera_and_pooled_queries_share_one_fit(monkeypatch, soils):
    photos, sample_ids, features, curves = soils
    calls = []
    select = nested.select_svr_c
    selections = []

    def observe_selection(x, y):
        result = select(x, y)
        selections.append((x.copy(), y.copy(), result))
        return result

    def observe_fit(x, y, query, *, C):
        calls.append((x.copy(), y.copy(), query.copy(), C))
        return svr_curves(x, y, query, C=C)

    monkeypatch.setattr(nested, "select_svr_c", observe_selection)
    monkeypatch.setattr(nested, "svr_curves", observe_fit)
    oof, cameras, selection_rows = nested.evaluate_nested_svr(photos, sample_ids, curves)
    outer_calls = [call for call in calls if len(call[2]) == 3]
    assert len(outer_calls) == len(selections) == len(sample_ids)
    for i, (x, y, query, C) in enumerate(outer_calls):
        keep = np.arange(len(features)) != i
        np.testing.assert_array_equal(x, features[keep])
        np.testing.assert_array_equal(y, curves[keep])
        np.testing.assert_array_equal(selections[i][0], x)
        np.testing.assert_array_equal(selections[i][1], y)
        assert selections[i][2][0] == C
        views = photos[photos["sample_id"] == sample_ids[i]].groupby("camera")[["feature_0", "feature_1"]].mean()
        np.testing.assert_array_equal(query, np.vstack([features[i:i + 1], views.to_numpy()]))
        expected = svr_curves(x, y, query, C=C)
        np.testing.assert_allclose(oof[i], expected[0])
        actual_views = cameras[cameras["sample_id"] == sample_ids[i]]
        np.testing.assert_allclose(actual_views[list(CANONICAL_GRAIN_LABELS)], expected[1:])
        np.testing.assert_allclose(actual_views["emd"], [emd_score(curves[i], p) for p in expected[1:]])
        fold_selection = selection_rows[selection_rows["sample_id"] == sample_ids[i]]
        assert fold_selection.loc[fold_selection["selected"], "C"].tolist() == [C]
        assert fold_selection.set_index("C")["inner_loo_emd"].to_dict() == selections[i][2][1]


def test_held_out_label_and_features_cannot_select_their_own_penalty(soils):
    photos, sample_ids, _, curves = soils
    original, _, selection = nested.evaluate_nested_svr(photos, sample_ids, curves)
    changed_curves = curves.copy()
    changed_curves[0] = [0] * 10 + [100]
    changed, _, target_selection = nested.evaluate_nested_svr(photos, sample_ids, changed_curves)
    np.testing.assert_allclose(original[0], changed[0])
    own_fold = selection["sample_id"] == sample_ids[0]
    pd.testing.assert_frame_equal(selection[own_fold], target_selection[own_fold])

    changed_photos = photos.copy()
    changed_photos.loc[changed_photos["sample_id"] == sample_ids[0], "feature_0"] = 1e6
    _, _, query_selection = nested.evaluate_nested_svr(changed_photos, sample_ids, curves)
    pd.testing.assert_frame_equal(selection[own_fold], query_selection[own_fold])


def test_exact_score_ties_choose_smallest_c():
    curves = np.tile([25] * 10 + [100], (3, 1))
    chosen, scores = nested.select_svr_c(np.arange(3)[:, None], curves)
    assert len(scores) == 3
    assert set(scores.values()) == {0.0}
    assert chosen == 0.1


def test_nested_selection_requires_nonempty_inner_training(soils):
    photos, sample_ids, features, curves = soils
    with pytest.raises(ValueError, match="at least two"):
        nested.select_svr_c(features[:1], curves[:1])
    with pytest.raises(ValueError, match="at least three"):
        nested.evaluate_nested_svr(photos, sample_ids[:2], curves[:2])
