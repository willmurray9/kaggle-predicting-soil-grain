import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import soilgrain.boost_model as boost_model
import soilgrain.search_experiment as search
import soilgrain.search_inputs as search_inputs
from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.ridge import ridge_curves
from soilgrain.targets import curve_array


@pytest.fixture
def search_run(tmp_path, monkeypatch):
    rng = np.random.default_rng(73)
    train_ids, test_ids = [f'S{i}' for i in range(8)], ['T2', 'T1']
    truth = pd.DataFrame(np.column_stack([
        np.sort(rng.uniform(5, 95, size=(8, 10)), axis=1), np.full(8, 100.0),
    ]), columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0, 'sample_id', train_ids)
    sample = pd.DataFrame(np.zeros((2, 11)), columns=CANONICAL_GRAIN_LABELS)
    sample.insert(0, 'sample_id', test_ids)
    keys = pd.DataFrame({
        'sample_id': np.repeat(train_ids + test_ids, 2),
        'camera': ['Phone 1', 'Phone 2'] * 10,
        'split': ['train'] * 16 + ['test'] * 4,
    })
    photos = {}
    for name, count, dtype in [('spectral', 23, np.float64), ('dino', 6, np.float32)]:
        photos[name] = pd.concat([
            keys, pd.DataFrame(rng.normal(size=(20, count)).astype(dtype),
                               columns=[f'feature_{i}' for i in range(count)]),
        ], axis=1)
    curves = curve_array(truth)

    def spectral_reference():
        frame = photos['spectral']
        columns = [c for c in frame if c.startswith('feature_')]
        train = frame.loc[frame['split'] == 'train']
        oof, camera = evaluate_samples(train, train_ids, curves, ridge_curves)
        predictions = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
        predictions.insert(0, 'sample_id', train_ids)
        candidate = sample.copy()
        means = frame.groupby('sample_id')[columns].mean()
        candidate[list(CANONICAL_GRAIN_LABELS)] = ridge_curves(
            means.loc[train_ids].to_numpy(), curves, means.loc[test_ids].to_numpy(),
        )
        return {'oof': predictions, 'camera': camera, 'submission': candidate}

    components = {'spectral': spectral_reference()}
    for name, offset in [('dino', 3.0), ('mobilenet', 7.0)]:
        components[name] = {}
        for kind, frame in components['spectral'].items():
            shifted = frame.copy(deep=True)
            shifted[list(CANONICAL_GRAIN_LABELS)[:10]] = np.clip(curve_array(frame)[:, :10] + offset, 0, 100)
            components[name][kind] = shifted
    artifacts = tmp_path / 'artifacts'
    submissions = artifacts / 'submissions'
    submissions.mkdir(parents=True)
    prior = submissions / 'prior.csv'
    components['spectral']['submission'].to_csv(prior, index=False)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'paths': {
        'artifacts_dir': str(artifacts), 'submissions_dir': str(submissions),
    }}))
    monkeypatch.setattr(search, 'PRIOR_SUBMISSIONS', ('prior.csv',))
    monkeypatch.setattr(search_inputs, 'load_search_inputs',
                        lambda _: (truth, sample, photos, components, []))
    monkeypatch.setattr(boost_model, 'boost_curves',
                        lambda train, targets, query: np.tile(targets.mean(axis=0), (len(query), 1)))
    return config, truth, sample, photos, components, spectral_reference


def test_search_driver_preserves_inputs_records_outputs_and_excludes_test_features(search_run):
    config, truth, sample, photos, components, spectral_reference = search_run
    inputs = [truth, sample, *photos.values(), *(frame for group in components.values() for frame in group.values())]
    snapshots = [frame.copy(deep=True) for frame in inputs]

    paths = search.write_search_experiment(config)
    summary = json.loads(paths['summary'].read_text())

    for actual, before in zip(inputs, snapshots, strict=True):
        pd.testing.assert_frame_equal(actual, before)
    expected_names = {'dino_pca', 'mobilenet_pca', 'spectral_dino_blend', 'spectral_mobilenet_blend',
                      'dino_pls', 'spectral_nested_ridge', 'spectral_boost'}
    assert {row['experiment'] for row in summary['candidates']} == expected_names
    assert summary['failed_candidates'] == []
    assert summary['train_samples'] == 8 and summary['test_samples'] == 2
    assert max(summary['reference_max_abs_differences'].values()) < 1e-10
    ranking = pd.read_csv(paths['ranking'])
    pd.testing.assert_frame_equal(ranking, ranking.sort_values(
        ['loo_emd', 'paired_camera_disagreement_emd', 'experiment'], ignore_index=True))
    skipped = {row['experiment'] for row in summary['duplicate_candidates']}
    assert summary['submission_queue'] == [name for name in ranking['experiment'] if name not in skipped]
    before_outputs = {}
    before_submissions = {}
    for row in summary['candidates']:
        for kind in ('oof', 'camera', 'submission'):
            record = row[kind]
            assert sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
        oof, camera, candidate = [pd.read_csv(row[kind]['path']) for kind in ('oof', 'camera', 'submission')]
        assert oof['sample_id'].tolist() == truth['sample_id'].tolist()
        assert len(camera) == 16 and row['camera_pairs'] == 8
        assert candidate['sample_id'].tolist() == sample['sample_id'].tolist()
        before_submissions[row['experiment']] = curve_array(candidate)
        kinds = ['oof', 'camera']
        if row['experiment'] in ('dino_pls', 'spectral_nested_ridge'):
            kinds += ['outer_selection', 'final_selection']
            for kind, count in [('outer_selection', 24), ('final_selection', 3)]:
                record = row[kind]
                assert sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
                assert len(pd.read_csv(record['path'])) == count
        before_outputs[row['experiment']] = {kind: Path(row[kind]['path']).read_bytes() for kind in kinds}
    for name in ('dino', 'mobilenet'):
        blend = next(row for row in summary['candidates'] if row['experiment'] == f'spectral_{name}_blend')
        np.testing.assert_allclose(curve_array(pd.read_csv(blend['oof']['path'])),
                                   (curve_array(components['spectral']['oof']) + curve_array(components[name]['oof'])) / 2)

    for frame in photos.values():
        columns = [c for c in frame if c.startswith('feature_')]
        frame.loc[frame['split'] == 'test', columns] *= -4
    components['spectral'] = spectral_reference()
    changed_inputs = [truth, sample, *photos.values(), *(frame for group in components.values() for frame in group.values())]
    changed_snapshots = [frame.copy(deep=True) for frame in changed_inputs]
    repeated = json.loads(search.write_search_experiment(config)['summary'].read_text())

    for actual, before in zip(changed_inputs, changed_snapshots, strict=True):
        pd.testing.assert_frame_equal(actual, before)
    for row in repeated['candidates']:
        assert all(Path(row[kind]['path']).read_bytes() == content
                   for kind, content in before_outputs[row['experiment']].items())
    for name in ('dino_pls', 'spectral_nested_ridge'):
        row = next(row for row in repeated['candidates'] if row['experiment'] == name)
        assert not np.allclose(curve_array(pd.read_csv(row['submission']['path'])), before_submissions[name])


def test_search_driver_records_numerical_exclusion_and_finishes_other_candidates(search_run, monkeypatch):
    config, *_ = search_run

    def fail_boost(*_args):
        raise FloatingPointError('synthetic nonfinite boost prediction')

    monkeypatch.setattr(boost_model, 'boost_curves', fail_boost)
    summary = json.loads(search.write_search_experiment(config)['summary'].read_text())

    assert summary['failed_candidates'] == [{'experiment': 'spectral_boost', 'error': 'synthetic nonfinite boost prediction'}]
    assert len(summary['candidates']) == 6
    assert 'spectral_boost' not in summary['submission_queue']


def test_search_blend_preserves_competition_column_spelling(search_run):
    config, _, sample, _, components, _ = search_run
    names = {'2.0': '2', '20.0': '20', '63.0': '63', '200.0': '200'}
    sample.rename(columns=names, inplace=True)
    for component in components.values():
        component['submission'].rename(columns=names, inplace=True)
    prior = config.parent / 'artifacts/submissions/prior.csv'
    components['spectral']['submission'].to_csv(prior, index=False)

    summary = json.loads(search.write_search_experiment(config)['summary'].read_text())

    for candidate in summary['candidates']:
        assert pd.read_csv(candidate['submission']['path']).columns.tolist() == sample.columns.tolist()
