import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.official_experiment import recipe_submission_evidence, write_official_experiment


@pytest.mark.parametrize('gain,camera,incumbent_gain,eligible', [
    (1., 10., 2., True),
    (0.99, 10., 2., False),
    (2., 10.01, 3., False),
    (2., 10., 0.5, False),
    (2., None, 3., False),
])
def test_recipe_screen_requires_both_incumbent_and_matched_improvement(gain, camera, incumbent_gain, eligible):
    errors = np.full(24, 40.)
    evidence = recipe_submission_evidence(errors, errors + incumbent_gain, camera, 29., errors + gain, 10.)
    assert evidence['eligible'] is eligible
    assert evidence['legacy_mean_improvement_emd'] == pytest.approx(gain)


@pytest.fixture
def official_run(tmp_path, monkeypatch):
    curated, artifacts = tmp_path / 'curated', tmp_path / 'artifacts'
    frozen = artifacts / 'experiments' / 'frozen_resnet18'
    pca = artifacts / 'experiments' / 'pca_ridge'
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
    features = np.random.default_rng(41).normal(size=(len(index), 512)).astype(np.float32)
    photos = pd.concat([index, pd.DataFrame(features, columns=[f'feature_{i}' for i in range(512)])], axis=1)
    photos.to_csv(frozen / 'photo_features.csv', index=False)
    (frozen / 'summary.json').write_text(json.dumps({
        'encoder': {'checkpoint_sha256': 'checkpoint'},
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
        (directory / 'summary.json').write_text('{}')
    curves(['T1', 'T2'], 35.).to_csv(submissions / 'ridge_rgb_texture_100.csv', index=False)

    def extract(actual_index, ppm, *, preprocessing):
        pd.testing.assert_frame_equal(actual_index[index.columns], index)
        assert preprocessing == 'official'
        return photos.copy(), {'checkpoint_sha256': 'checkpoint'}

    monkeypatch.setattr('soilgrain.official_experiment.extract_frozen_features', extract)
    return config, artifacts, photos


def test_official_run_aligns_references_preserves_inputs_and_records_decision(official_run):
    config, artifacts, photos = official_run
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = write_official_experiment(config)
    summary = json.loads(paths['summary'].read_text())
    assert summary['final_alpha'] == 1000.
    assert summary['selected_submission'] == str(paths['pca_submission'])
    assert len(pd.read_csv(paths['outer_selection'])) == 33
    assert all(p.read_bytes() == content for p, content in before.items())
    for key, value in (('pca_submission', 50.), ('blend_submission', 42.5)):
        submission = pd.read_csv(paths[key])
        assert submission['sample_id'].tolist() == ['T2', 'T1']
        np.testing.assert_allclose(submission['0.002'], value)
    comparison = pd.read_csv(paths['comparison'])
    assert comparison['sample_id'].tolist() == [f'S{i:02}' for i in range(11)][::-1]
    np.testing.assert_allclose(comparison['emd_incumbent'], 75.)
    np.testing.assert_allclose(comparison['emd_legacy_pca'], 25.)
    np.testing.assert_allclose(comparison['emd_legacy_blend'], 50.)
    np.testing.assert_allclose(comparison['emd_pca'], 0.)
    np.testing.assert_allclose(comparison['emd_blend'], 37.5)
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*summary['sources'], *summary['photo_sources'], *summary['submissions']])
    # Changing test features cannot affect the training selection or OOF outputs.
    retained = {key: paths[key].read_bytes() for key in ('oof', 'camera_predictions', 'outer_selection', 'final_selection')}
    photos.loc[photos['split'] == 'test', [c for c in photos if c.startswith('feature_')]] = 1e6
    write_official_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())


@pytest.mark.parametrize('corruption', ['photo', 'camera', 'calibration'])
def test_official_run_rejects_changed_photos_or_reference_coverage(official_run, corruption):
    config, artifacts, photos = official_run
    if corruption == 'photo':
        Path(photos.iloc[0]['path']).write_bytes(b'changed')
    elif corruption == 'calibration':
        (config.parent / 'curated' / 'ppm.csv').write_text('phone,ppm\none,2.\ntwo,2.\n')
    else:
        path = artifacts / 'experiments' / 'pca_ridge' / 'camera_predictions.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'camera'] = 'wrong'
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='photo|coverage|source changed'):
        write_official_experiment(config)
    assert not (artifacts / 'experiments' / 'official_preprocessing').exists()
