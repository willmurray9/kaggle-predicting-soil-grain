import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.physical_photo_experiment import write_physical_photo_experiment
from test_physical_photo_experiment import physical_photo_run


def run(config):
    from soilgrain.spectral_photo_experiment import write_spectral_photo_experiment
    return write_spectral_photo_experiment(config)


def update_spectrum_hash(artifacts):
    directory = artifacts / 'experiments' / 'physical_photo'
    path = directory / 'summary.json'
    summary = json.loads(path.read_text())
    summary['feature_cache']['sha256'] = sha256((directory / 'photo_spectrum_features.csv').read_bytes()).hexdigest()
    path.write_text(json.dumps(summary))


def test_combination_reconstructs_new_reference_preserves_inputs_and_aligns_caches(physical_photo_run):
    config, artifacts, _, _ = physical_photo_run
    prior = write_physical_photo_experiment(config)
    spectrum = pd.read_csv(prior['features'])
    spectrum.iloc[::-1].to_csv(prior['features'], index=False)
    update_spectrum_hash(artifacts)
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = run(config)
    summary = json.loads(paths['summary'].read_text())
    assert summary['feature_count'] == 23
    assert summary['reference_experiment'] == 'spectral_ridge'
    assert max(summary['reference_max_abs_differences'].values()) < 1e-10
    assert all(p.read_bytes() == content for p, content in before.items())
    assert pd.read_csv(paths['submission'])['sample_id'].tolist() == ['T2', 'T1']
    assert len(pd.read_csv(paths['oof'])) == 6
    assert len(pd.read_csv(paths['camera_predictions'])) == 12
    prior_summary = json.loads(prior['summary'].read_text())
    reference = next(row for row in prior_summary['experiments'] if row['experiment'] == 'spectral_ridge')
    assert summary['incumbent_emd'] == pytest.approx(reference['loo_emd'])
    assert summary['incumbent_camera_disagreement_emd'] == pytest.approx(reference['paired_camera_disagreement_emd'])
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*summary['sources'], *summary['photo_sources'], summary['submission']])
    assert summary['selected_submission'] == (str(paths['submission']) if
        summary['loo_emd'] < summary['incumbent_emd'] and
        summary['paired_camera_disagreement_emd'] < summary['incumbent_camera_disagreement_emd'] else None)


@pytest.mark.parametrize('corruption', ['photo', 'spectrum', 'rgb', 'oof', 'camera', 'submission'])
def test_combination_rejects_changed_sources_and_reference(physical_photo_run, corruption):
    config, artifacts, photos, _ = physical_photo_run
    prior = write_physical_photo_experiment(config)
    if corruption == 'photo':
        Path(photos.iloc[0]['path']).write_bytes(b'changed')
    else:
        path = {'spectrum': prior['features'],
                'rgb': artifacts / 'experiments' / 'nested_texture_kernel' / 'photo_features.csv',
                'oof': prior['oof'], 'camera': prior['camera_predictions'],
                'submission': prior['spectral_ridge_submission']}[corruption]
        frame = pd.read_csv(path)
        column = 'spectrum_0' if corruption == 'spectrum' else 'feature_0' if corruption == 'rgb' else '0.002'
        frame.loc[0, column] -= 0.01
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='source changed|reference'):
        run(config)
    assert not (artifacts / 'experiments' / 'spectral_photo').exists()


@pytest.mark.parametrize('corruption', ['duplicate', 'camera', 'nonfinite'])
def test_combination_rejects_misaligned_spectrum_even_with_updated_hash(physical_photo_run, corruption):
    config, artifacts, _, _ = physical_photo_run
    prior = write_physical_photo_experiment(config)
    frame = pd.read_csv(prior['features'])
    if corruption == 'duplicate':
        frame.iloc[0] = frame.iloc[1]
    elif corruption == 'camera':
        frame.loc[0, 'camera'] = 'unknown'
    else:
        frame.loc[0, 'spectrum_0'] = np.nan
    frame.to_csv(prior['features'], index=False)
    update_spectrum_hash(artifacts)
    with pytest.raises(ValueError, match='Spectral cache'):
        run(config)
    assert not (artifacts / 'experiments' / 'spectral_photo').exists()


def test_combination_validation_does_not_depend_on_test_features(physical_photo_run):
    config, artifacts, photos, spectrum = physical_photo_run
    write_physical_photo_experiment(config)
    paths = run(config)
    before = {key: paths[key].read_bytes() for key in ('oof', 'camera_predictions', 'comparison')}
    for path in photos.loc[photos['split'] == 'test', 'path']:
        spectrum[path] = np.array([1., 0., 0., 0., 0., 0.])
    write_physical_photo_experiment(config)
    run(config)
    assert all(paths[key].read_bytes() == content for key, content in before.items())
