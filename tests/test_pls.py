import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

import soilgrain.pls as pls
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score


def test_one_component_recovers_linear_curves_with_an_unpenalized_intercept():
    features = np.array([[-2, 7], [0, 7], [2, 7]], dtype=float)
    base = np.arange(5, 100, 10, dtype=float)
    curves = np.column_stack([base + 2 * features[:, :1], np.full(3, 100)])

    predictions = pls.pls_curves(features, curves, np.array([[-1, 7], [1, 7]]), n_components=1)

    np.testing.assert_allclose(predictions[:, :10], [base - 2, base + 2], atol=1e-10)
    np.testing.assert_array_equal(predictions[:, -1], 100)


def test_supervised_component_finds_signal_amid_correlated_nuisance_features():
    signal = np.tile([-1, 1], 4)
    nuisance = np.tile([-1, -1, 1, 1], 2)
    features = np.column_stack([signal, nuisance, nuisance, nuisance])
    curves = np.column_stack([np.repeat((40 + 5 * signal)[:, None], 10, axis=1), np.full(8, 100)])

    predictions = pls.pls_curves(features, curves, np.array([[-1, 0, 0, 0], [1, 0, 0, 0]]), n_components=1)

    # The dominant feature-variance direction is nuisance; only signal covaries with targets.
    np.testing.assert_allclose(predictions[:, :10], np.repeat([[35], [45]], 10, axis=1), atol=1e-10)


@pytest.fixture
def soils():
    rng = np.random.default_rng(4)
    ids = ["H", "B", "G", "A", "F", "D", "E", "C"]
    columns = [f"feature_{i}" for i in range(5)]
    rows = []
    for sample_id, values in zip(ids, rng.normal(size=(8, 5)), strict=True):
        for camera, offset in [("a", -0.25), ("a", 0.25), ("b", 0.5)]:
            rows.append({"sample_id": sample_id, "camera": camera, **dict(zip(columns, values + offset))})
    photos = pd.DataFrame(rows)
    features = photos.astype({c: np.float32 for c in columns}).groupby("sample_id")[columns].mean().loc[ids].to_numpy()
    curves = np.column_stack([np.sort(rng.uniform(10, 80, (8, 10)), axis=1), np.full(8, 100)])
    return photos, ids, features, curves


def test_feature_units_target_translation_and_query_cohort_do_not_change_fit(soils):
    _, _, features, curves = soils
    query = features[:2].astype(float) + 0.1
    original = pls.pls_curves(features, curves, query, n_components=2)
    multipliers = np.array([2, 0.01, 10, 0.5, 3])
    converted = pls.pls_curves(features * multipliers + 7, curves, query * multipliers + 7, n_components=2)
    batched = pls.pls_curves(features, curves, np.vstack([query, np.full(5, 1e9)]), n_components=2)
    shifted = curves.copy()
    shifted[:, :10] += 2
    translated = pls.pls_curves(features, shifted, query, n_components=2)

    np.testing.assert_allclose(converted, original, atol=1e-9)
    np.testing.assert_allclose(batched[:2], original, atol=1e-9)
    np.testing.assert_allclose(translated[:, :10], original[:, :10] + 2, atol=1e-9)
    assert np.isfinite(batched).all() and np.all(np.diff(batched, axis=1) >= 0)
    assert np.all((batched >= 0) & (batched <= 100))


@pytest.mark.parametrize("constant", [1.0, 0.1])
def test_constant_features_or_targets_use_training_mean_and_existing_curve_repair(constant):
    curves = np.tile([-10, 20, 10, 30, 110, 90, 80, 70, 60, 50, 100], (3, 1))
    expected = [[0, 20, 20, 30, 100, 100, 100, 100, 100, 100, 100]]
    np.testing.assert_allclose(pls.pls_curves(np.arange(3)[:, None], curves, [[10]], n_components=1), expected)
    curves = np.array([[10] * 10 + [100], [40] * 10 + [100], [70] * 10 + [100]])
    np.testing.assert_allclose(pls.pls_curves(np.full((3, 2), constant), curves, [[100, -100]], n_components=1), [[40] * 10 + [100]])


def test_empty_query_preserves_output_width(soils):
    _, _, features, curves = soils
    assert pls.pls_curves(features, curves, np.empty((0, 5)), n_components=1).shape == (0, 11)


@pytest.mark.parametrize("components", [0, -1, True, 1.5, 8])
def test_invalid_components_are_rejected(soils, components):
    _, _, features, curves = soils
    with pytest.raises(ValueError, match="n_components"):
        pls.pls_curves(features, curves, features[:1], n_components=components)


@pytest.mark.parametrize("features,curves,query", [
    (np.zeros(2), np.zeros((2, 11)), np.zeros((1, 1))),
    (np.zeros((0, 1)), np.zeros((0, 11)), np.zeros((1, 1))),
    (np.zeros((2, 1)), np.zeros((2, 10)), np.zeros((1, 1))),
    (np.zeros((2, 1)), np.zeros((2, 11)), np.zeros((1, 2))),
    (np.zeros((2, 1)), np.zeros((2, 11)), np.full((1, 1), np.nan)),
])
def test_misaligned_or_nonfinite_arrays_are_rejected(features, curves, query):
    with pytest.raises(ValueError):
        pls.pls_curves(features, curves, query, n_components=1)


@pytest.mark.parametrize("failure", ["warning", "nonfinite"])
def test_solver_failure_is_not_silently_replaced(monkeypatch, failure):
    class FailedSolver:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            if failure == "warning":
                warnings.warn("iteration limit", ConvergenceWarning)
            return self

        def predict(self, query):
            return np.full((len(query), 10), np.nan)

    monkeypatch.setattr(pls, "PLSRegression", FailedSolver)
    with pytest.raises((FloatingPointError, ConvergenceWarning)):
        pls.pls_curves([[-1], [1]], [[40] * 10 + [100], [60] * 10 + [100]], [[0]], n_components=1)


def test_inner_selection_excludes_each_soil_and_scores_its_predictions(monkeypatch, soils):
    _, _, features, curves = soils
    original_fit = pls.pls_curves
    calls = []

    def observe(x, y, query, *, n_components):
        prediction = original_fit(x, y, query, n_components=n_components)
        calls.append((x.copy(), y.copy(), query.copy(), n_components, prediction))
        return prediction

    monkeypatch.setattr(pls, "pls_curves", observe)
    selected, scores = pls.select_pls_components(features, curves)

    assert len(calls) == 3 * len(features)
    for components in (1, 2, 4):
        fits = [call for call in calls if call[3] == components]
        for i, (x, y, query, _, _) in enumerate(fits):
            keep = np.arange(len(features)) != i
            np.testing.assert_array_equal(x, features[keep])
            np.testing.assert_array_equal(y, curves[keep])
            np.testing.assert_array_equal(query, features[i:i + 1])
        assert scores[components] == emd_score(curves, np.vstack([call[4] for call in fits]))
    assert selected == min(scores, key=lambda k: (scores[k], k))


def test_outer_pooled_and_camera_queries_share_training_only_fit(monkeypatch, soils):
    photos, ids, features, curves = soils
    original_fit, original_select = pls.pls_curves, pls.select_pls_components
    outer_calls, selections = [], []

    def observe_fit(x, y, query, *, n_components):
        prediction = original_fit(x, y, query, n_components=n_components)
        if len(query) == 3:
            outer_calls.append((x.copy(), y.copy(), query.copy(), n_components, prediction))
        return prediction

    def observe_select(x, y):
        result = original_select(x, y)
        selections.append((x.copy(), y.copy(), result))
        return result

    monkeypatch.setattr(pls, "pls_curves", observe_fit)
    monkeypatch.setattr(pls, "select_pls_components", observe_select)
    oof, views, selection = pls.evaluate_nested_pls(photos, ids, curves)

    assert len(outer_calls) == len(selections) == len(ids)
    columns = [c for c in photos if c.startswith("feature_")]
    for i, (x, y, query, components, prediction) in enumerate(outer_calls):
        keep = np.arange(len(ids)) != i
        np.testing.assert_array_equal(x, features[keep])
        np.testing.assert_array_equal(y, curves[keep])
        np.testing.assert_array_equal(selections[i][0], x)
        np.testing.assert_array_equal(selections[i][1], y)
        assert selections[i][2][0] == components
        camera_means = photos[photos.sample_id == ids[i]].astype({c: np.float32 for c in columns}).groupby("camera")[columns].mean()
        np.testing.assert_array_equal(query, np.vstack([features[i:i + 1], camera_means.to_numpy()]))
        np.testing.assert_allclose(oof[i], prediction[0])
        np.testing.assert_allclose(views.loc[views.sample_id == ids[i], list(CANONICAL_GRAIN_LABELS)], prediction[1:])
        own_selection = selection[selection.sample_id == ids[i]]
        assert own_selection.loc[own_selection.selected, "n_components"].tolist() == [components]
        assert own_selection.set_index("n_components").inner_loo_emd.to_dict() == selections[i][2][1]


def test_held_out_labels_and_photos_cannot_choose_their_own_components(soils):
    photos, ids, _, curves = soils
    original, _, selection = pls.evaluate_nested_pls(photos, ids, curves)
    changed_curves = curves.copy()
    changed_curves[0] = [0] * 10 + [100]
    changed, _, label_selection = pls.evaluate_nested_pls(photos, ids, changed_curves)
    np.testing.assert_allclose(original[0], changed[0], atol=1e-12)
    own_fold = selection.sample_id == ids[0]
    pd.testing.assert_frame_equal(selection[own_fold], label_selection[own_fold])
    changed_photos = photos.copy()
    changed_photos.loc[changed_photos.sample_id == ids[0], "feature_0"] = 1e5
    _, _, photo_selection = pls.evaluate_nested_pls(changed_photos, ids, curves)
    pd.testing.assert_frame_equal(selection[own_fold], photo_selection[own_fold])


def test_exact_selection_ties_choose_fewer_components(soils):
    _, _, features, _ = soils
    selected, scores = pls.select_pls_components(features, np.tile([25] * 10 + [100], (8, 1)))
    assert scores == {1: 0.0, 2: 0.0, 4: 0.0}
    assert selected == 1


def test_nested_selection_requires_enough_training_rows_for_four_components(soils):
    photos, ids, features, curves = soils
    with pytest.raises(ValueError):
        pls.select_pls_components(features[:5], curves[:5])
    with pytest.raises(ValueError):
        pls.evaluate_nested_pls(photos[photos.sample_id.isin(ids[:6])], ids[:6], curves[:6])


@pytest.mark.parametrize("problem", ["duplicate_id", "extra_soil", "missing_camera", "nonfinite", "test_split"])
def test_nested_evaluation_rejects_ambiguous_or_leaking_photo_rows(soils, problem):
    photos, ids, _, curves = soils
    photos, ids = photos.copy(), ids.copy()
    if problem == "duplicate_id":
        ids[0] = ids[1]
    elif problem == "extra_soil":
        photos.loc[0, "sample_id"] = "unknown"
    elif problem == "missing_camera":
        photos.loc[0, "camera"] = None
    elif problem == "nonfinite":
        photos.loc[0, "feature_0"] = np.inf
    else:
        photos["split"] = "test"
    with pytest.raises(ValueError):
        pls.evaluate_nested_pls(photos, ids, curves)
