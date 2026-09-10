import numpy as np
import pandas as pd
import pytest

import soilgrain.nested_neighbors as nested
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.metrics import emd_score
from soilgrain.multicrop import COMPONENTS


@pytest.fixture
def soils():
    sample_ids = ["I", "B", "F", "A", "H", "D", "C", "G", "E"]
    rng = np.random.default_rng(41)
    features = {name: rng.normal(size=(9, 2)) for name in COMPONENTS}
    curves = np.array([list(np.linspace(v, 90, 10)) + [100] for v in [2, 30, 12, 55, 65, 8, 45, 20, 75]])
    photos = {}
    for name in COMPONENTS:
        photos[name] = pd.DataFrame([
            {"sample_id": sample_id, "camera": camera,
             "feature_0": values[0] + offset, "feature_1": values[1] - offset}
            for sample_id, values in zip(sample_ids, features[name], strict=True)
            for camera, offset in [("phone_a", -0.25), ("phone_b", 0.25)]
        ])
    return photos, sample_ids, features, curves


def manual_blend(features, curves, queries, count):
    predictions = []
    for name in COMPONENTS:
        scale = np.std(features[name], axis=0)
        scale[scale < 1e-12] = 1
        crop_predictions = []
        for query in queries[name]:
            distances = [sum(((query - row) / scale) ** 2) for row in features[name]]
            neighbors = sorted(range(len(distances)), key=lambda i: distances[i])[:count]
            crop_predictions.append(np.mean(curves[neighbors], axis=0))
        predictions.append(crop_predictions)
    return np.mean(predictions, axis=0)


def manual_selection(features, curves):
    scores = {}
    for count in (1, 3, 5, 7):
        predictions = []
        for i in range(len(curves)):
            keep = np.arange(len(curves)) != i
            predictions.append(manual_blend(
                {name: values[keep] for name, values in features.items()}, curves[keep],
                {name: values[i:i + 1] for name, values in features.items()}, count,
            )[0])
        scores[count] = emd_score(curves, np.array(predictions))
    return min(scores, key=lambda count: (scores[count], -count)), scores


def test_nested_predictions_match_independent_inner_scaling_and_camera_oracle(soils):
    photos, sample_ids, features, curves = soils
    # Crop file order must not determine which camera curves get averaged.
    photos[COMPONENTS[1]] = photos[COMPONENTS[1]].iloc[::-1]
    oof, cameras, selections = nested.evaluate_nested_neighbors(photos, sample_ids, curves)
    for i, sample_id in enumerate(sample_ids):
        keep = np.arange(len(curves)) != i
        training = {name: values[keep] for name, values in features.items()}
        chosen, scores = manual_selection(training, curves[keep])
        selection = selections[selections["sample_id"] == sample_id]
        assert selection.loc[selection["selected"], "n_neighbors"].tolist() == [chosen]
        assert selection.set_index("n_neighbors")["inner_loo_emd"].to_dict() == pytest.approx(scores)
        expected = manual_blend(training, curves[keep], {
            name: values[i:i + 1] for name, values in features.items()
        }, chosen)
        np.testing.assert_allclose(oof[i], expected[0])
        for row in cameras[cameras["sample_id"] == sample_id].itertuples(index=False, name=None):
            _, camera, error, *prediction = row
            queries = {
                name: frame.loc[(frame["sample_id"] == sample_id) & (frame["camera"] == camera),
                                ["feature_0", "feature_1"]].to_numpy()
                for name, frame in photos.items()
            }
            expected = manual_blend(training, curves[keep], queries, chosen)[0]
            np.testing.assert_allclose(prediction, expected)
            assert error == pytest.approx(emd_score(curves[i], expected))
    assert cameras.columns.tolist() == ["sample_id", "camera", "emd", *CANONICAL_GRAIN_LABELS]


def test_held_out_labels_and_queries_cannot_choose_own_count(soils):
    photos, sample_ids, _, curves = soils
    original, _, selections = nested.evaluate_nested_neighbors(photos, sample_ids, curves)
    changed_curves = curves.copy()
    changed_curves[0] = [0] * 10 + [100]
    changed, _, changed_selections = nested.evaluate_nested_neighbors(photos, sample_ids, changed_curves)
    own_fold = selections["sample_id"] == sample_ids[0]
    np.testing.assert_array_equal(original[0], changed[0])
    pd.testing.assert_frame_equal(selections[own_fold], changed_selections[own_fold])
    changed_photos = {name: frame.copy() for name, frame in photos.items()}
    for frame in changed_photos.values():
        frame.loc[frame["sample_id"] == sample_ids[0], "feature_0"] = 1e6
    _, _, query_selections = nested.evaluate_nested_neighbors(changed_photos, sample_ids, curves)
    pd.testing.assert_frame_equal(selections[own_fold], query_selections[own_fold])


def test_camera_views_use_one_shared_selection_per_soil(monkeypatch, soils):
    photos, sample_ids, _, curves = soils
    calls = []
    select = nested.select_neighbor_count

    def observe(features, targets):
        calls.append(len(targets))
        return select(features, targets)

    monkeypatch.setattr(nested, "select_neighbor_count", observe)
    original, _, selections = nested.evaluate_nested_neighbors(photos, sample_ids, curves)
    assert calls == [8] * 9
    changed_photos = {name: frame.copy() for name, frame in photos.items()}
    for frame in changed_photos.values():
        # Preserve each soil's pooled features while moving the camera queries.
        frame["feature_0"] += np.tile([-10, 10], len(sample_ids))
    changed, _, changed_selections = nested.evaluate_nested_neighbors(changed_photos, sample_ids, curves)
    assert calls == [8] * 18
    np.testing.assert_allclose(original, changed)
    pd.testing.assert_frame_equal(selections, changed_selections)


def test_exact_ties_prefer_more_neighbors():
    features = {name: np.arange(8, dtype=float)[:, None] for name in COMPONENTS}
    curves = np.tile([25] * 10 + [100], (8, 1))
    chosen, scores = nested.select_neighbor_count(features, curves)
    assert scores == {1: 0.0, 3: 0.0, 5: 0.0, 7: 0.0}
    assert chosen == 7


def test_multicrop_prediction_scales_from_training_only(soils):
    _, _, features, curves = soils
    queries = {name: values[:1] for name, values in features.items()}
    expected = manual_blend(features, curves, queries, 5)
    alone = nested.predict_multicrop_neighbors(features, curves, queries, n_neighbors=5)
    together = nested.predict_multicrop_neighbors(features, curves, {
        name: np.vstack([values, [1e9, -1e9]]) for name, values in queries.items()
    }, n_neighbors=5)
    np.testing.assert_allclose(alone, expected)
    np.testing.assert_array_equal(alone[0], together[0])


def test_grid_requires_enough_inner_training_soils(soils):
    photos, sample_ids, features, curves = soils
    with pytest.raises(ValueError, match="at least eight"):
        nested.select_neighbor_count({name: values[:7] for name, values in features.items()}, curves[:7])
    with pytest.raises(ValueError, match="at least nine"):
        nested.evaluate_nested_neighbors(photos, sample_ids[:8], curves[:8])


def test_crop_and_camera_coverage_must_match(soils):
    photos, sample_ids, features, curves = soils
    with pytest.raises(ValueError, match="components"):
        nested.select_neighbor_count({"gray_50": features["gray_50"]}, curves)
    changed = dict(photos)
    changed[COMPONENTS[0]] = changed[COMPONENTS[0]].iloc[1:]
    with pytest.raises(ValueError, match="camera.*coverage"):
        nested.evaluate_nested_neighbors(changed, sample_ids, curves)
