import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.physical_photo_experiment import write_physical_photo_experiment
from soilgrain.spectral_photo_experiment import write_spectral_photo_experiment
from test_physical_photo_experiment import physical_photo_run


@pytest.fixture
def geometry_run(physical_photo_run, monkeypatch):
    config, artifacts, photos, spectrum = physical_photo_run
    write_physical_photo_experiment(config)
    write_spectral_photo_experiment(config)
    rng = np.random.default_rng(43)
    morphology = {p: np.concatenate([rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))]) for p in photos['path']}
    from soilgrain import geometry_transport_experiment
    monkeypatch.setattr(geometry_transport_experiment, 'physical_granulometry_features', lambda path, camera: morphology[path])
    return config, artifacts, photos, spectrum, morphology


def test_geometry_transport_preserves_inputs_reference_and_test_independence(geometry_run):
    from soilgrain.geometry_transport_experiment import write_geometry_transport_experiment
    config, artifacts, photos, spectrum, morphology = geometry_run
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = write_geometry_transport_experiment(config)
    s = json.loads(paths['summary'].read_text())
    assert max(s['reference_max_abs_differences'].values()) < 1e-10
    assert [row['feature_count'] for row in s['experiments']] == [12, 23]
    assert s['roundtrip_max_abs_percentage_points'] <= .05 + 1e-9
    assert s['roundtrip_mean_emd'] <= .25 + 1e-9
    assert all(p.read_bytes() == value for p, value in before.items())
    assert len(pd.read_csv(paths['oof'])) == 12
    assert len(pd.read_csv(paths['camera_predictions'])) == 24
    for name in ('granulometry_ridge', 'transport_ridge'):
        assert pd.read_csv(paths[f'{name}_submission'])['sample_id'].tolist() == ['T2', 'T1']
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*s['sources'], *s['photo_sources'], *s['submissions'], s['feature_cache']])
    retained = {key: paths[key].read_bytes() for key in ('oof', 'camera_predictions', 'comparison')}
    for p in photos.loc[photos['split'] == 'test', 'path']:
        spectrum[p] = np.array([1., 0., 0., 0., 0., 0.])
        morphology[p] = np.tile(spectrum[p], 2)
    write_physical_photo_experiment(config)
    write_spectral_photo_experiment(config)
    write_geometry_transport_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())


@pytest.mark.parametrize('corruption', ['photo', 'spectrum', 'reference', 'fingerprint'])
def test_geometry_transport_rejects_source_corruption_before_new_extraction(geometry_run, monkeypatch, corruption):
    from soilgrain.geometry_transport_experiment import write_geometry_transport_experiment
    config, artifacts, photos, _, _ = geometry_run
    if corruption == 'photo':
        Path(photos.iloc[0]['path']).write_bytes(b'changed')
    elif corruption == 'fingerprint':
        path = artifacts / 'experiments' / 'spectral_photo' / 'summary.json'
        s = json.loads(path.read_text()); s['sources'] = s['sources'][1:]; path.write_text(json.dumps(s))
    else:
        path = artifacts / 'experiments' / 'physical_photo' / ('photo_spectrum_features.csv' if corruption == 'spectrum' else 'camera_predictions.csv')
        frame = pd.read_csv(path); frame.loc[0, 'spectrum_0' if corruption == 'spectrum' else '0.002'] -= .01
        frame.to_csv(path, index=False)
    monkeypatch.setattr('soilgrain.geometry_transport_experiment.physical_granulometry_features', lambda *_: pytest.fail('New extraction ran before provenance checks'))
    with pytest.raises(ValueError, match='source|fingerprint'):
        write_geometry_transport_experiment(config)
    assert not (artifacts / 'experiments' / 'geometry_transport').exists()


@pytest.mark.parametrize('morphology,transport,expected', [
    ((39., 24.), (38., 23.), 'transport_ridge'),
    ((39., 24.), (39., 23.), 'granulometry_ridge'),
    ((40., 24.), (39., 25.), None),
    ((39., None), (41., 24.), None),
])
def test_fixed_choice_requires_dual_gain_and_tie_prefers_granulometry(morphology, transport, expected):
    from soilgrain.geometry_transport_experiment import _select_candidate
    rows = [{'experiment': name, 'loo_emd': error, 'paired_camera_disagreement_emd': camera}
            for name, (error, camera) in [('granulometry_ridge', morphology), ('transport_ridge', transport)]]
    assert _select_candidate(rows, 40., 25.) == expected
