import numpy as np
import pytest


def test_equal_blend_fits_each_feature_group_independently():
    from soilgrain.native_followup_experiment import native_lbp_curves

    features = np.zeros((3, 63))
    features[:, 0] = [-1, 0, 1]
    features[:, 23] = [-1, 0, 1]
    curves = np.column_stack([np.repeat([[10], [40], [70]], 10, axis=1), [100]*3])
    query = np.zeros((2, 63))
    query[0, [0, 23]] = [2, -1]
    query[1, [0, 23]] = [-1, 2]
    # Each independent fit has slope 90/13; averaging gives 40 + 45/13.
    actual = native_lbp_curves(features, curves, query)
    np.testing.assert_allclose(actual[:, :10], 43.46153846153846)
    np.testing.assert_array_equal(actual[:, 10], [100, 100])


def test_blend_query_batch_cannot_change_fitted_scaling():
    from soilgrain.native_followup_experiment import native_lbp_curves

    rng = np.random.default_rng(42)
    features = rng.normal(size=(8, 63))
    curves = np.sort(rng.uniform(0, 100, (8, 11)), axis=1)
    curves[:, -1] = 100
    query = rng.normal(size=(1, 63))
    separate = native_lbp_curves(features, curves, query)
    together = native_lbp_curves(features, curves, np.vstack([query, np.full((1, 63), 1e6)]))
    np.testing.assert_allclose(together[:1], separate)


@pytest.mark.parametrize('train_columns,query_columns', [(62, 63), (64, 63), (63, 62)])
def test_blend_rejects_wrong_feature_partition(train_columns, query_columns):
    from soilgrain.native_followup_experiment import native_lbp_curves

    with pytest.raises(ValueError):
        native_lbp_curves(np.zeros((3, train_columns)), np.full((3, 11), 100.),
                          np.zeros((1, query_columns)))


def test_runner_aligns_photo_caches_and_preserves_previous_results(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import pandas as pd

    from soilgrain import native_followup_experiment as experiment
    from soilgrain.constants import CANONICAL_GRAIN_LABELS
    from soilgrain.distribution_experiment import record

    source = tmp_path / 'experiments' / 'distribution_search'
    source.mkdir(parents=True)
    keys, native_values, lbp_values = [], [], []
    for i, value in enumerate([-1., 0., 1., 2.]):
        for camera in ['Motorola Edge', 'Samsung A52']:
            keys.append(['train', f'S{i}', camera, f'{i}_{camera}'])
            native_values.append([value] + [0.]*22)
            lbp_values.append([-10*value] + [0.]*39)
    for soil, value in [('T2', 2.), ('T1', -1.)]:
        keys.append(['test', soil, 'iPhone', soil])
        native_values.append([value] + [0.]*22)
        lbp_values.append([-10*value] + [0.]*39)
    index = pd.DataFrame(keys, columns=['split', 'sample_id', 'camera', 'path'])
    records = []
    for name, values in [('native_spectral', native_values), ('lbp', lbp_values)]:
        features = pd.DataFrame(values, columns=[f'feature_{i}' for i in range(len(values[0]))])
        frame = pd.concat([index, features], axis=1)
        path = source / f'{name}_photo_features.csv'
        frame.iloc[::-1].to_csv(path, index=False)
        records.append(record(path))
    truth = pd.DataFrame([[v]*10+[100.] for v in [10., 30., 50., 70.]],
                         columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0, 'sample_id', ['S0', 'S1', 'S2', 'S3'])
    reference_path = source / 'reference.csv'
    truth.iloc[::-1].to_csv(reference_path, index=False)
    summary = source / 'summary.json'
    summary.write_text(json.dumps({'sources': records, 'candidates': [
        {'experiment': 'native_spectral_ridge', 'oof': record(reference_path)}]}))
    sample = truth.iloc[:2].copy()
    sample['sample_id'] = ['T2', 'T1']
    before = {str(p): p.read_bytes() for p in source.iterdir()}
    monkeypatch.setattr(experiment, 'PINNED_DISTRIBUTION', record(summary)['sha256'])
    monkeypatch.setattr(experiment, 'load_config', lambda _: SimpleNamespace(artifacts_dir=tmp_path))
    monkeypatch.setattr(experiment, 'load_search_inputs',
                        lambda _: (truth, sample, {'spectral': index}, {}, []))
    result = experiment.run_native_followup()
    blend = next(c for c in result['candidates'] if c['experiment'] == 'native_lbp_blend')
    submission = pd.read_csv(blend['submission']['path'])
    assert submission.sample_id.tolist() == ['T2', 'T1']
    np.testing.assert_allclose(submission.iloc[:, 1:11],
                               np.repeat([[48.57142857142857], [31.42857142857143]], 10, axis=1))
    transfer = pd.read_csv(blend['camera_transfer']['path'])
    assert len(transfer) == 16
    for row in transfer.itertuples():
        assert set(json.loads(row.training_soil_ids)) == set(truth.sample_id) - {row.sample_id}
    for path, contents in before.items():
        from pathlib import Path
        assert Path(path).read_bytes() == contents
    # Re-running must never overwrite accepted or audited batch artifacts.
    with pytest.raises(FileExistsError):
        experiment.run_native_followup()
