import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.mobilenet_experiment import write_mobilenet_experiment


@pytest.fixture
def mobilenet_run(tmp_path, monkeypatch):
    curated, artifacts = tmp_path / 'curated', tmp_path / 'artifacts'
    frozen = artifacts / 'experiments' / 'frozen_resnet18'
    pca = artifacts / 'experiments' / 'official_preprocessing'
    texture = artifacts / 'experiments' / 'physical_texture'
    reports, submissions = artifacts / 'reports', artifacts / 'submissions'
    for directory in (curated, frozen, pca, texture, reports, submissions):
        directory.mkdir(parents=True)
    config = tmp_path / 'config.yaml'
    config.write_text(
        f'paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n  reports_dir: {reports}\n  submissions_dir: {submissions}\n'
        'files:\n  train: train.csv\n  sample_submission: sample.csv\n  ppm: ppm.csv\n'
    )
    ids = [f'S{i:02}' for i in range(11)][::-1]

    def curves(soil_ids, value):
        frame = pd.DataFrame({'sample_id': soil_ids})
        frame[list(CANONICAL_GRAIN_LABELS)] = [[value] * 10 + [100.]] * len(soil_ids)
        return frame

    curves(ids, 50.).to_csv(curated / 'train.csv', index=False)
    curves(['T2', 'T1'], 0.).to_csv(curated / 'sample.csv', index=False)
    pd.DataFrame({'phone': ['one', 'two'], 'ppm': [1., 1.]}).to_csv(curated / 'ppm.csv', index=False)
    index = pd.DataFrame([
        {'sample_id': soil, 'split': 'test' if soil.startswith('T') else 'train',
         'camera': camera, 'path': str(tmp_path / f'{soil}_{camera}.jpg')}
        for soil in [*ids, 'T1', 'T2'] for camera in ('one', 'two')
    ])
    for path in index['path']:
        Path(path).write_bytes(b'photo fixture')
    index.to_csv(reports / 'photo_index.csv', index=False)
    features = np.random.default_rng(41).normal(size=(len(index), 960)).astype(np.float32)
    photos = pd.concat([index, pd.DataFrame(features, columns=[f'feature_{i}' for i in range(960)])], axis=1)
    photos.to_csv(frozen / 'photo_features.csv', index=False)
    (pca / 'summary.json').write_text(json.dumps({
        'sources': [{'path': str(curated / 'ppm.csv'), 'sha256': sha256((curated / 'ppm.csv').read_bytes()).hexdigest()}],
        'photo_sources': [{'path': p, 'sha256': sha256(b'photo fixture').hexdigest()} for p in index['path']],
    }))
    for directory, configurations in ((pca, [('pca', 45.), ('blend', 40.)]),
                                       (texture, [('ridge_rgb_texture_100', 35.)])):
        oof, cameras = [], []
        for name, value in configurations:
            frame = curves(ids[::-1], value)
            frame.insert(0, 'experiment', name)
            frame['emd'] = 999.  # Recompute errors from curves, never trust stored scores.
            oof.append(frame)
            views = curves([soil for soil in ids[::-1] for _ in range(2)], value)
            views.insert(1, 'camera', ['two', 'one'] * len(ids))
            views.insert(0, 'experiment', name)
            cameras.append(views)
        pd.concat(oof).to_csv(directory / 'oof_predictions.csv', index=False)
        pd.concat(cameras).to_csv(directory / 'camera_predictions.csv', index=False)
        if directory == texture:
            (directory / 'summary.json').write_text('{}')
    curves(['T1', 'T2'], 35.).to_csv(submissions / 'ridge_rgb_texture_100.csv', index=False)

    fallback = submissions / 'rgb_texture_official_pca8_blend.csv'
    curves(['T2', 'T1'], 40.).to_csv(fallback, index=False)
    manifest_path = pca / 'summary.json'
    manifest = json.loads(manifest_path.read_text())
    reference_paths = [pca / f'{kind}_predictions.csv' for kind in ('oof', 'camera')]
    manifest['sources'].extend({'path': str(p), 'sha256': sha256(p.read_bytes()).hexdigest()} for p in reference_paths)
    manifest['references'] = {'incumbent': {'outer_loo_emd': 75., 'paired_camera_disagreement_emd': 0.}}
    manifest['experiments'] = [{'experiment': name, 'outer_loo_emd': score, 'paired_camera_disagreement_emd': 0.} for name, score in [('pca', 25.), ('blend', 50.)]]
    manifest['submissions'] = [{'path': str(fallback), 'sha256': sha256(fallback.read_bytes()).hexdigest()}]
    manifest['feature_cache'] = {'path': str(frozen / 'photo_features.csv'), 'sha256': sha256((frozen / 'photo_features.csv').read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))

    def extract(actual_index, ppm, *, preprocessing, encoder_name):
        pd.testing.assert_frame_equal(actual_index[index.columns], index)
        assert preprocessing == 'official'
        assert encoder_name == 'mobilenet_v3_large'
        return photos.copy(), {'checkpoint_sha256': 'checkpoint'}

    monkeypatch.setattr('soilgrain.mobilenet_experiment.extract_frozen_features', extract)
    return config, artifacts, photos


def test_mobilenet_run_preserves_inputs_and_keeps_test_out_of_selection(mobilenet_run):
    config, artifacts, photos = mobilenet_run
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = write_mobilenet_experiment(config)
    summary = json.loads(paths['summary'].read_text())
    assert summary['input_features'] == 960
    assert summary['pca_components'] == 8
    assert summary['final_alpha'] == 1000.
    assert summary['selected_submission'] == str(paths['submission'])
    assert len(pd.read_csv(paths['outer_selection'])) == 33
    candidate = pd.read_csv(paths['submission'])
    assert candidate['sample_id'].tolist() == ['T2', 'T1']
    np.testing.assert_allclose(candidate['0.002'], 50.)
    comparison = pd.read_csv(paths['comparison'])
    np.testing.assert_allclose(comparison['emd_incumbent'], 75.)
    np.testing.assert_allclose(comparison['emd_official_pca'], 25.)
    np.testing.assert_allclose(comparison['emd_fallback_blend'], 50.)
    np.testing.assert_allclose(comparison['emd_mobilenet'], 0.)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*summary['sources'], *summary['photo_sources'], summary['submission'], summary['feature_cache']])
    retained = {key: paths[key].read_bytes() for key in ('oof', 'camera_predictions', 'outer_selection', 'final_selection')}
    photos.loc[photos['split'] == 'test', [c for c in photos if c.startswith('feature_')]] = 1e6
    write_mobilenet_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())


@pytest.mark.parametrize('prediction,selected', [(50., 'mobilenet'), (40., 'fallback'), (30., 'fallback')])
def test_informative_selection_uses_lower_emd_and_ties_prefer_fallback(mobilenet_run, monkeypatch, prediction, selected):
    config, artifacts, _photos = mobilenet_run
    from soilgrain.nested_ridge import evaluate_nested_ridge

    def controlled_predictions(photos, ids, curves, **kwargs):
        oof, views, selection = evaluate_nested_ridge(photos, ids, curves, **kwargs)
        oof[:, :10] = prediction
        views[list(CANONICAL_GRAIN_LABELS)[:10]] = prediction
        return oof, views, selection

    monkeypatch.setattr('soilgrain.mobilenet_experiment.evaluate_nested_ridge', controlled_predictions)
    paths = write_mobilenet_experiment(config)
    summary = json.loads(paths['summary'].read_text())
    expected = paths['submission'] if selected == 'mobilenet' else artifacts / 'submissions' / 'rgb_texture_official_pca8_blend.csv'
    assert summary['selected_submission'] == str(expected)
    if prediction == 30.:
        assert not summary['incumbent_screen_diagnostic']['eligible']


@pytest.mark.parametrize('corruption', ['photo', 'camera', 'calibration', 'candidate'])
def test_mobilenet_run_rejects_changed_sources_before_extraction(mobilenet_run, corruption):
    config, artifacts, photos = mobilenet_run
    if corruption == 'photo':
        Path(photos.iloc[0]['path']).write_bytes(b'changed')
    elif corruption == 'calibration':
        (config.parent / 'curated' / 'ppm.csv').write_text('phone,ppm\none,2.\ntwo,2.\n')
    else:
        path = (artifacts / 'submissions' / 'rgb_texture_official_pca8_blend.csv' if corruption == 'candidate'
                else artifacts / 'experiments' / 'official_preprocessing' / 'camera_predictions.csv')
        frame = pd.read_csv(path)
        frame.loc[0, '0.002' if corruption == 'candidate' else 'camera'] = 1. if corruption == 'candidate' else 'wrong'
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='photo|coverage|source changed'):
        write_mobilenet_experiment(config)
    assert not (artifacts / 'experiments' / 'mobilenet_pca').exists()
