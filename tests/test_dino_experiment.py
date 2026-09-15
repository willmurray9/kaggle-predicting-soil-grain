import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.multicrop import _score_rows
from soilgrain.pca_experiment import _camera_disagreement


@pytest.fixture
def dino_run(tmp_path, monkeypatch):
    curated, artifacts = tmp_path / 'curated', tmp_path / 'artifacts'
    reference_dir = artifacts / 'experiments' / 'physical_photo'
    manifest_path = artifacts / 'experiments' / 'spectral_photo' / 'summary.json'
    reports, submissions = artifacts / 'reports', artifacts / 'submissions'
    for directory in (curated, reference_dir, manifest_path.parent, reports, submissions):
        directory.mkdir(parents=True)
    config = tmp_path / 'config.yaml'
    config.write_text(
        f'paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n  reports_dir: {reports}\n  submissions_dir: {submissions}\n'
        'files:\n  train: train.csv\n  sample_submission: sample.csv\n  ppm: ppm.csv\n'
    )
    ids = [f'S{i:02}' for i in range(11)][::-1]
    truth = pd.DataFrame({'sample_id': ids})
    truth[list(CANONICAL_GRAIN_LABELS)] = [[float(v)] * 10 + [100.] for v in np.linspace(10., 90., 11)]
    truth.to_csv(curated / 'train.csv', index=False)
    sample = truth.iloc[:2].copy(); sample['sample_id'] = ['T2', 'T1']
    sample.to_csv(curated / 'sample.csv', index=False)
    pd.DataFrame({'phone': ['one', 'two'], 'ppm': [1., 1.]}).to_csv(curated / 'ppm.csv', index=False)
    index = pd.DataFrame([
        {'split': 'test' if soil.startswith('T') else 'train', 'sample_id': soil,
         'camera': camera, 'path': str(tmp_path / f'{soil}_{camera}.jpg')}
        for soil in [*ids, 'T1', 'T2'] for camera in ('one', 'two')
    ])
    for path in index['path']:
        Path(path).write_bytes(b'photo fixture')
    index.to_csv(reports / 'photo_index.csv', index=False)
    oof = truth.copy(); oof[list(CANONICAL_GRAIN_LABELS)[:10]] = 35.
    oof.insert(0, 'experiment', 'spectral_ridge'); oof['emd'] = 999.
    views = index.loc[index['split'] == 'train', ['sample_id', 'camera']].copy()
    views[list(CANONICAL_GRAIN_LABELS)] = [[30. if camera == 'one' else 40.] * 10 + [100.] for camera in views['camera']]
    views.insert(0, 'experiment', 'spectral_ridge'); views['emd'] = 999.
    oof.iloc[::-1].to_csv(reference_dir / 'oof_predictions.csv', index=False)
    views.iloc[::-1].to_csv(reference_dir / 'camera_predictions.csv', index=False)
    candidate_path = submissions / 'rgb_texture_spectral_ridge.csv'
    sample.iloc[::-1].to_csv(candidate_path, index=False)
    source_paths = [config, *(curated / name for name in ('train.csv', 'sample.csv', 'ppm.csv')),
                    reports / 'photo_index.csv', reference_dir / 'oof_predictions.csv', reference_dir / 'camera_predictions.csv', candidate_path]
    manifest_path.write_text(json.dumps({
        'sources': [{'path': str(p), 'sha256': sha256(p.read_bytes()).hexdigest()} for p in source_paths],
        'photo_sources': [{'path': p, 'sha256': sha256(Path(p).read_bytes()).hexdigest()} for p in index['path']],
        'incumbent_emd': float(_score_rows(truth, oof).mean()),
        'incumbent_camera_disagreement_emd': _camera_disagreement(views)[0],
    }))
    photos = pd.concat([index, pd.DataFrame(np.random.default_rng(37).normal(size=(len(index), 384)).astype(np.float32),
                                           columns=[f'feature_{i}' for i in range(384)])], axis=1)
    from soilgrain import dino_experiment
    monkeypatch.setattr(dino_experiment, 'extract_dino_features', lambda actual_index, ppm: (photos.copy(), {'checkpoint_sha256': 'fixture'}))
    return config, artifacts, photos


def test_dino_pipeline_uses_spectral_reference_and_excludes_test_from_nested_selection(dino_run):
    from soilgrain.dino_experiment import write_dino_experiment
    config, artifacts, photos = dino_run
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = write_dino_experiment(config)
    s = json.loads(paths['summary'].read_text())
    assert all(p.read_bytes() == value for p, value in before.items())
    assert s['reference_experiment'] == 'spectral_ridge'
    assert s['input_features'] == 384 and s['pca_components'] == 8
    assert s['ridge_alphas'] == [10., 100., 1000.]
    assert len(pd.read_csv(paths['outer_selection'])) == 33
    assert len(pd.read_csv(paths['oof'])) == 11
    assert len(pd.read_csv(paths['camera_predictions'])) == 22
    assert pd.read_csv(paths['submission'])['sample_id'].tolist() == ['T2', 'T1']
    assert s['selected_submission'] == (str(paths['submission']) if s['outer_loo_emd'] < s['incumbent_emd']
        and s['paired_camera_disagreement_emd'] < s['incumbent_camera_disagreement_emd'] else None)
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*s['sources'], *s['photo_sources'], s['feature_cache'], s['submission']])
    retained = {key: paths[key].read_bytes() for key in ('oof', 'camera_predictions', 'comparison', 'outer_selection', 'final_selection')}
    photos.loc[photos['split'] == 'test', [c for c in photos if c.startswith('feature_')]] = 1e6
    write_dino_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())


@pytest.mark.parametrize('corruption', ['photo', 'calibration', 'reference', 'fingerprint'])
def test_dino_pipeline_rejects_changed_sources_before_extraction(dino_run, monkeypatch, corruption):
    from soilgrain.dino_experiment import write_dino_experiment
    config, artifacts, photos = dino_run
    if corruption == 'photo':
        Path(photos.iloc[0]['path']).write_bytes(b'changed')
    elif corruption == 'calibration':
        (config.parent / 'curated' / 'ppm.csv').write_text('phone,ppm\none,2\ntwo,2\n')
    elif corruption == 'reference':
        path = artifacts / 'experiments' / 'physical_photo' / 'camera_predictions.csv'
        frame = pd.read_csv(path); frame.loc[0, 'camera'] = 'wrong'; frame.to_csv(path, index=False)
    else:
        path = artifacts / 'experiments' / 'spectral_photo' / 'summary.json'
        s = json.loads(path.read_text()); s['sources'] = s['sources'][:-1]; path.write_text(json.dumps(s))
    monkeypatch.setattr('soilgrain.dino_experiment.extract_dino_features', lambda *_: pytest.fail('Extraction ran before source checks'))
    with pytest.raises(ValueError, match='source|fingerprint'):
        write_dino_experiment(config)
    assert not (artifacts / 'experiments' / 'dino_pca').exists()


@pytest.mark.parametrize('corruption', ['keys', 'nonfinite', 'dimension'])
def test_dino_pipeline_rejects_invalid_extracted_features(dino_run, corruption):
    from soilgrain.dino_experiment import write_dino_experiment
    config, artifacts, photos = dino_run
    if corruption == 'keys':
        photos.loc[0, 'sample_id'] = 'wrong'
    elif corruption == 'nonfinite':
        photos.loc[0, 'feature_0'] = np.nan
    else:
        photos.drop(columns=['feature_383'], inplace=True)
    with pytest.raises(ValueError, match='Extracted features'):
        write_dino_experiment(config)
    assert not (artifacts / 'experiments' / 'dino_pca').exists()
