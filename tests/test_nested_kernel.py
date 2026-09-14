import numpy as np
import pandas as pd
import pytest

import soilgrain.nested_kernel as nested
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.kernel_ridge import kernel_ridge_curves
from soilgrain.metrics import emd_score


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

    def observe(x, y, query, *, alpha, gamma):
        calls.append((x.copy(), y.copy(), query.copy(), alpha, gamma))
        return kernel_ridge_curves(x, y, query, alpha=alpha, gamma=gamma)

    monkeypatch.setattr(nested, "kernel_ridge_curves", observe)
    chosen, scores = nested.select_kernel_parameters(features, curves)

    assert len(calls) == len(features) * 6
    expected = {}
    for alpha in (0.1, 1.0, 10.0):
        for gamma_scale in (0.1, 1.0):
            gamma = gamma_scale / 2
            predictions = []
            setting_calls = [call for call in calls if call[3:] == (alpha, gamma)]
            assert len(setting_calls) == len(features)
            for i, (x, y, query, _, _) in enumerate(setting_calls):
                keep = np.arange(len(features)) != i
                np.testing.assert_array_equal(x, features[keep])
                np.testing.assert_array_equal(y, curves[keep])
                np.testing.assert_array_equal(query, features[i:i + 1])
                predictions.append(kernel_ridge_curves(x, y, query, alpha=alpha, gamma=gamma)[0])
            expected[(alpha, gamma_scale)] = emd_score(curves, np.vstack(predictions))
    assert scores == expected
    assert chosen == min(expected, key=lambda pair: (expected[pair], -pair[0], pair[1]))


def test_outer_predictions_match_manual_nested_loop(soils):
    photos, sample_ids, features, curves = soils
    oof, cameras, selections = nested.evaluate_nested_kernel(photos, sample_ids, curves)

    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(features)) != i
        inner_features, inner_curves = features[keep], curves[keep]
        scores = {}
        for alpha in (0.1, 1.0, 10.0):
            for gamma_scale in (0.1, 1.0):
                inner_predictions = []
                for j in range(len(inner_features)):
                    inner_keep = np.arange(len(inner_features)) != j
                    inner_predictions.append(kernel_ridge_curves(
                        inner_features[inner_keep], inner_curves[inner_keep],
                        inner_features[j:j + 1], alpha=alpha, gamma=gamma_scale / 2,
                    )[0])
                scores[(alpha, gamma_scale)] = emd_score(inner_curves, np.vstack(inner_predictions))
        chosen = min(scores, key=lambda pair: (scores[pair], -pair[0], pair[1]))
        selection = selections[selections["sample_id"] == sample_id]
        assert selection.loc[selection["selected"], ["alpha", "gamma_scale"]].values.tolist() == [list(chosen)]
        assert selection.set_index(["alpha", "gamma_scale"])["inner_loo_emd"].to_dict() == scores
        options = {"alpha": chosen[0], "gamma": chosen[1] / 2}
        expected = kernel_ridge_curves(features[keep], curves[keep], features[i:i + 1], **options)
        np.testing.assert_allclose(oof[i], expected[0])
        views = photos[photos["sample_id"] == sample_id].set_index("camera")
        for row in cameras[cameras["sample_id"] == sample_id].itertuples(index=False, name=None):
            _, camera, error, *prediction = row
            query = views.loc[[camera], ["feature_0", "feature_1"]].to_numpy()
            expected = kernel_ridge_curves(features[keep], curves[keep], query, **options)[0]
            np.testing.assert_allclose(prediction, expected)
            assert error == pytest.approx(emd_score(curves[i], expected))
    assert cameras.columns.tolist() == ["sample_id", "camera", "emd", *CANONICAL_GRAIN_LABELS]


def test_held_out_target_and_query_features_cannot_choose_own_parameters(soils):
    photos, sample_ids, _, curves = soils
    original, _, selection = nested.evaluate_nested_kernel(photos, sample_ids, curves)
    changed_curves = curves.copy()
    changed_curves[0] = [0] * 10 + [100]
    changed, _, target_selection = nested.evaluate_nested_kernel(photos, sample_ids, changed_curves)
    np.testing.assert_allclose(original[0], changed[0])
    own_fold = selection["sample_id"] == sample_ids[0]
    pd.testing.assert_frame_equal(selection[own_fold], target_selection[own_fold])

    changed_photos = photos.copy()
    changed_photos.loc[changed_photos["sample_id"] == sample_ids[0], "feature_0"] = 1e6
    _, _, query_selection = nested.evaluate_nested_kernel(changed_photos, sample_ids, curves)
    pd.testing.assert_frame_equal(selection[own_fold], query_selection[own_fold])


def test_camera_queries_share_one_selection_per_outer_fold(monkeypatch, soils):
    photos, sample_ids, _, curves = soils
    calls = []
    select = nested.select_kernel_parameters

    def observe(features, targets):
        calls.append(features.copy())
        return select(features, targets)

    monkeypatch.setattr(nested, "select_kernel_parameters", observe)
    original, _, selection = nested.evaluate_nested_kernel(photos, sample_ids, curves)
    assert len(calls) == len(sample_ids)
    changed_photos = photos.copy()
    changed_photos["feature_0"] += np.tile([-100, 100], len(sample_ids))
    changed, _, changed_selection = nested.evaluate_nested_kernel(changed_photos, sample_ids, curves)
    assert len(calls) == 2 * len(sample_ids)
    np.testing.assert_allclose(original, changed)
    pd.testing.assert_frame_equal(selection, changed_selection)


def test_exact_score_ties_choose_strongest_penalty_then_broadest_kernel():
    curves = np.tile([25] * 10 + [100], (3, 1))
    chosen, scores = nested.select_kernel_parameters(np.arange(3)[:, None], curves)
    assert len(scores) == 6
    assert set(scores.values()) == {0.0}
    assert chosen == (10.0, 0.1)


def test_selection_and_outer_evaluation_require_nonempty_inner_training(soils):
    photos, sample_ids, features, curves = soils
    with pytest.raises(ValueError, match="at least two"):
        nested.select_kernel_parameters(features[:1], curves[:1])
    with pytest.raises(ValueError, match="at least three"):
        nested.evaluate_nested_kernel(photos, sample_ids[:2], curves[:2])
