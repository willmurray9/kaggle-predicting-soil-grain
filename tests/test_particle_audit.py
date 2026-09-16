import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def audit_inputs(tmp_path):
    artifacts, curated = tmp_path / 'artifacts', tmp_path / 'curated'
    (artifacts / 'reports').mkdir(parents=True)
    (artifacts / 'experiments/patch_mixture').mkdir(parents=True)
    curated.mkdir()
    config = tmp_path / 'data.yaml'
    config.write_text(f'paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n  reports_dir: {artifacts}/reports\nfiles:\n  ppm: ppm.csv\n')
    calibration = curated / 'ppm.csv'
    pd.DataFrame({'phone': ['Motorola Edge', 'Samsung A52'], 'ppm': [2., 2.], 'width': [200, 200], 'height': [240, 240]}).to_csv(calibration, index=False)
    rows = []
    for soil in ('F827', 'H183', 'H516', 'H668'):
        for camera in ('Motorola Edge', 'Samsung A52'):
            path = tmp_path / f'{soil}_{camera}.png'
            rgb = np.zeros((240, 200, 3), dtype=np.uint8)
            rgb[70:150, 60:140] = 200
            Image.fromarray(rgb).save(path)
            rows.append({'split': 'train', 'sample_id': soil, 'camera': camera, 'path': str(path)})
    index = artifacts / 'reports/photo_index.csv'
    pd.DataFrame(rows[::-1]).to_csv(index, index=False)
    manifest = artifacts / 'experiments/patch_mixture/summary.json'
    manifest.write_text(json.dumps({
        'sources': [{'path': str(p), 'sha256': sha256(p.read_bytes()).hexdigest()} for p in (config, calibration, index)],
        'photo_sources': [{'path': row['path'], 'sha256': sha256(Path(row['path']).read_bytes()).hexdigest()} for row in rows],
    }))
    return config, artifacts, rows, index, calibration


def test_particle_audit_preserves_sources_and_records_calibrated_outputs(audit_inputs):
    from soilgrain.particle_audit import write_particle_audit
    config, artifacts, rows, _, _ = audit_inputs
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = write_particle_audit(config)
    summary = json.loads(paths['summary'].read_text())
    assert summary['scope'] == 'Image-only feasibility audit; no soil targets, test images, fitting or submission'
    assert summary['panel_soils'] == ['F827', 'H183', 'H516', 'H668']
    photos = pd.read_csv(paths['photos'])
    assert len(photos) == 8
    assert set(photos['source_path']) == {row['path'] for row in rows}
    assert set(photos['crop_pixels']) == {200}
    assert set(photos['effective_ppm']) == {2.}
    assert set(photos['crop_top']) == {20}
    assert set(photos['crop_left']) == {0}
    for record in summary['photos']:
        arrays = np.load(record['arrays_path'])
        assert arrays['labels'].shape == arrays['seeds'].shape == (200, 200)
        assert Image.open(record['crop_path']).size == (200, 200)
    assert len(summary['overlays']) == 4
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*summary['sources'], *summary['photo_sources'], *summary['outputs']])
    assert not (artifacts / 'submissions').exists()


@pytest.mark.parametrize('corruption', ['photo', 'index', 'calibration'])
def test_particle_audit_checks_sources_before_reading_images(audit_inputs, monkeypatch, corruption):
    from soilgrain.particle_audit import write_particle_audit
    config, artifacts, rows, index, calibration = audit_inputs
    path = {'photo': Path(rows[0]['path']), 'index': index, 'calibration': calibration}[corruption]
    path.write_bytes(path.read_bytes() + b'changed')
    monkeypatch.setattr('soilgrain.particle_audit.physical_source_crop', lambda *_: pytest.fail('Image processing preceded hash verification'))
    with pytest.raises(ValueError, match='source changed'):
        write_particle_audit(config)
    assert not (artifacts / 'experiments/particle_audit').exists()
