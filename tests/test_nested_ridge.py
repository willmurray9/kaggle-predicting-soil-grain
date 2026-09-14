import numpy as np
import pandas as pd
import pytest

import soilgrain.nested_ridge as nested
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.ridge import ridge_curves


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

    def observe(x, y, query, *, alpha):
        calls.append((x.copy(), y.copy(), query.copy(), alpha))
        return ridge_curves(x, y, query, alpha=alpha)

    monkeypatch.setattr(nested, "ridge_curves", observe)
    chosen, scores = nested.select_ridge_alpha(features, curves)

    assert len(calls) == len(features) * len(nested.RIDGE_ALPHAS)
    expected = {}
    for alpha in nested.RIDGE_ALPHAS:
        predictions = []
        alpha_calls = [call for call in calls if call[3] == alpha]
        for i, (x, y, query, _) in enumerate(alpha_calls):
            keep = np.arange(len(features)) != i
            np.testing.assert_array_equal(x, features[keep])
            np.testing.assert_array_equal(y, curves[keep])
            np.testing.assert_array_equal(query, features[i:i + 1])
            predictions.append(ridge_curves(features[keep], curves[keep], query, alpha=alpha)[0])
        expected[alpha] = emd_score(curves, np.vstack(predictions))
    assert scores == expected
    assert chosen == min(expected, key=lambda alpha: (expected[alpha], -alpha))


@pytest.mark.parametrize("n_components", [None, 1])
def test_outer_predictions_match_manual_nested_loop(soils, n_components):
    photos, sample_ids, features, curves = soils
    options = {} if n_components is None else {"n_components": n_components}
    oof, cameras, selections = nested.evaluate_nested_ridge(photos, sample_ids, curves, **options)

    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(features)) != i
        inner_features, inner_curves = features[keep], curves[keep]
        scores = {}
        for alpha in nested.RIDGE_ALPHAS:
            inner_predictions = []
            for j in range(len(inner_features)):
                inner_keep = np.arange(len(inner_features)) != j
                inner_predictions.append(ridge_curves(
                    inner_features[inner_keep], inner_curves[inner_keep],
                    inner_features[j:j + 1], alpha=alpha, **options,
                )[0])
            scores[alpha] = emd_score(inner_curves, np.vstack(inner_predictions))
        chosen = min(scores, key=lambda alpha: (scores[alpha], -alpha))
        selection = selections[selections["sample_id"] == sample_id]
        assert selection.loc[selection["selected"], "alpha"].tolist() == [chosen]
        assert selection.set_index("alpha")["inner_loo_emd"].to_dict() == scores
        expected = ridge_curves(features[keep], curves[keep], features[i:i + 1], alpha=chosen, **options)
        np.testing.assert_allclose(oof[i], expected[0])
        views = photos[photos["sample_id"] == sample_id].set_index("camera")
        for row in cameras[cameras["sample_id"] == sample_id].itertuples(index=False, name=None):
            _, camera, error, *prediction = row
            query = views.loc[[camera], ["feature_0", "feature_1"]].to_numpy()
            expected = ridge_curves(features[keep], curves[keep], query, alpha=chosen, **options)[0]
            np.testing.assert_allclose(prediction, expected)
            assert error == pytest.approx(emd_score(curves[i], expected))
    assert cameras.columns.tolist() == ["sample_id", "camera", "emd", *CANONICAL_GRAIN_LABELS]


@pytest.mark.parametrize("n_components", [None, 1])
def test_held_out_target_and_query_features_cannot_choose_own_alpha(soils, n_components):
    photos, sample_ids, _, curves = soils
    options = {} if n_components is None else {"n_components": n_components}
    original, _, selection = nested.evaluate_nested_ridge(photos, sample_ids, curves, **options)
    changed_curves = curves.copy()
    changed_curves[0] = [0] * 10 + [100]
    changed, _, target_selection = nested.evaluate_nested_ridge(photos, sample_ids, changed_curves, **options)
    np.testing.assert_allclose(original[0], changed[0])
    own_fold = selection["sample_id"] == sample_ids[0]
    pd.testing.assert_frame_equal(selection[own_fold], target_selection[own_fold])

    changed_photos = photos.copy()
    changed_photos.loc[changed_photos["sample_id"] == sample_ids[0], "feature_0"] = 1e6
    _, _, query_selection = nested.evaluate_nested_ridge(changed_photos, sample_ids, curves, **options)
    pd.testing.assert_frame_equal(selection[own_fold], query_selection[own_fold])


def test_camera_queries_share_one_selection_per_outer_fold(monkeypatch, soils):
    photos, sample_ids, _, curves = soils
    calls = []
    select = nested.select_ridge_alpha

    def observe(features, targets):
        calls.append(features.copy())
        return select(features, targets)

    monkeypatch.setattr(nested, "select_ridge_alpha", observe)
    original, _, selection = nested.evaluate_nested_ridge(photos, sample_ids, curves)
    assert len(calls) == len(sample_ids)
    changed_photos = photos.copy()
    changed_photos["feature_0"] += np.tile([-100, 100], len(sample_ids))
    changed, _, changed_selection = nested.evaluate_nested_ridge(changed_photos, sample_ids, curves)
    assert len(calls) == 2 * len(sample_ids)
    np.testing.assert_allclose(original, changed)
    pd.testing.assert_frame_equal(selection, changed_selection)


def test_exact_score_ties_choose_strongest_penalty():
    curves = np.tile([25] * 10 + [100], (3, 1))
    chosen, scores = nested.select_ridge_alpha(np.arange(3)[:, None], curves)
    assert scores == {10.0: 0.0, 100.0: 0.0, 1000.0: 0.0}
    assert chosen == 1000.0


def test_selection_and_outer_evaluation_require_nonempty_inner_training(soils):
    photos, sample_ids, features, curves = soils
    with pytest.raises(ValueError, match="at least two"):
        nested.select_ridge_alpha(features[:1], curves[:1])
    with pytest.raises(ValueError, match="at least three"):
        nested.evaluate_nested_ridge(photos, sample_ids[:2], curves[:2])


def test_pca_is_refitted_on_each_inner_training_set(monkeypatch, soils):
    _, _, features, curves = soils
    svd = np.linalg.svd
    fitted = []

    def observe(values, **kwargs):
        fitted.append(values.copy())
        return svd(values, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", observe)
    nested.select_ridge_alpha(features, curves, n_components=1)

    assert len(fitted) == len(features) * len(nested.RIDGE_ALPHAS)
    for i, values in enumerate(fitted):
        training = features[np.arange(len(features)) != i % len(features)]
        np.testing.assert_allclose(values, (training - training.mean(axis=0)) / training.std(axis=0))


def test_nested_pca_must_fit_inside_inner_training_capacity(soils):
    photos, sample_ids, _, curves = soils
    with pytest.raises(ValueError, match="n_components"):
        nested.evaluate_nested_ridge(photos, sample_ids[:3], curves[:3], n_components=1)
