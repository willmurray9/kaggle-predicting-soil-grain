import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.geometry_transport_experiment import write_geometry_transport_experiment
from test_geometry_transport_experiment import geometry_run
from test_physical_photo_experiment import physical_photo_run


@pytest.fixture
def patch_run(geometry_run, monkeypatch):
    config, artifacts, photos, _, _ = geometry_run
    write_geometry_transport_experiment(config)
    rng = np.random.default_rng(73)
    spreads = {path: rng.uniform(0, .2, 6) for path in photos['path']}
    from soilgrain import patch_experiment
    monkeypatch.setattr(patch_experiment, 'physical_patch_iqr_features', lambda path, camera: spreads[path])
    return config, artifacts, photos, spreads


def test_patch_run_preserves_reference_features_and_excludes_test_photos(patch_run):
    from soilgrain.patch_experiment import write_patch_experiment
    config, artifacts, photos, spreads = patch_run
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = write_patch_experiment(config)
    summary = json.loads(paths['summary'].read_text())
    assert summary['feature_count'] == 29
    assert max(summary['reference_max_abs_differences'].values()) < 1e-10
    assert all(p.read_bytes() == value for p, value in before.items())
    features = pd.read_csv(paths['features']).set_index('path').loc[photos['path']]
    original = [f'feature_{i}' for i in range(17)]
    np.testing.assert_allclose(features[original], photos[original], rtol=0, atol=1e-15)
    spectral = pd.read_csv(artifacts / 'experiments/physical_photo/photo_spectrum_features.csv').set_index('path').loc[photos['path']]
    np.testing.assert_allclose(features[[f'feature_{i}' for i in range(17, 23)]], spectral[[f'spectrum_{i}' for i in range(6)]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(features[[f'feature_{i}' for i in range(23, 29)]], np.vstack([spreads[p] for p in photos['path']]), rtol=0, atol=1e-15)
    assert len(pd.read_csv(paths['oof'])) == 6
    assert len(pd.read_csv(paths['camera_predictions'])) == 12
    assert pd.read_csv(paths['submission'])['sample_id'].tolist() == ['T2', 'T1']
    expected = summary['loo_emd'] < summary['incumbent_emd'] and summary['paired_camera_disagreement_emd'] < summary['incumbent_camera_disagreement_emd']
    assert (summary['selected_submission'] is not None) == expected
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*summary['sources'], *summary['photo_sources'], summary['submission'], summary['feature_cache']])
    retained = {key: paths[key].read_bytes() for key in ('oof', 'camera_predictions', 'comparison')}
    old_submission = paths['submission'].read_bytes()
    for path in photos.loc[photos['split'] == 'test', 'path']:
        spreads[path] = np.full(6, .95)
    write_patch_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())
    assert paths['submission'].read_bytes() != old_submission


@pytest.mark.parametrize('corruption', ['photo', 'spectrum', 'fingerprint', 'reference_with_updated_hash'])
def test_patch_rejects_corruption_before_extraction(patch_run, monkeypatch, corruption):
    from soilgrain.patch_experiment import write_patch_experiment
    config, artifacts, photos, _ = patch_run
    manifest_path = artifacts / 'experiments/geometry_transport/summary.json'
    manifest = json.loads(manifest_path.read_text())
    if corruption == 'photo':
        Path(photos.iloc[0]['path']).write_bytes(b'changed')
    elif corruption == 'fingerprint':
        manifest['sources'] = manifest['sources'][1:]
        manifest_path.write_text(json.dumps(manifest))
    else:
        path = artifacts / 'experiments/physical_photo' / ('photo_spectrum_features.csv' if corruption == 'spectrum' else 'camera_predictions.csv')
        frame = pd.read_csv(path)
        frame.loc[0, 'spectrum_0' if corruption == 'spectrum' else '0.002'] -= .01
        frame.to_csv(path, index=False)
        if corruption == 'reference_with_updated_hash':
            for item in manifest['sources']:
                if Path(item['path']) == path:
                    item['sha256'] = sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr('soilgrain.patch_experiment.physical_patch_iqr_features', lambda *_: pytest.fail('Extraction ran before provenance/reconstruction checks'))
    with pytest.raises(ValueError, match='source|fingerprint|reconstruct'):
        write_patch_experiment(config)
    assert not (artifacts / 'experiments/patch_mixture').exists()


def test_patch_rejects_invalid_extractor_output(patch_run, monkeypatch):
    from soilgrain.patch_experiment import write_patch_experiment
    config, artifacts, _, _ = patch_run
    monkeypatch.setattr('soilgrain.patch_experiment.physical_patch_iqr_features', lambda *_: np.full(6, np.nan))
    with pytest.raises(ValueError, match='six finite'):
        write_patch_experiment(config)
    assert not (artifacts / 'experiments/patch_mixture').exists()
