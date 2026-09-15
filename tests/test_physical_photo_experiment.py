import json
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.experiments import evaluate_samples
from soilgrain.ridge import ridge_curves
from soilgrain.physical_photo_experiment import _select_candidate, write_physical_photo_experiment


@pytest.fixture
def physical_photo_run(tmp_path, monkeypatch):
    curated, artifacts = tmp_path / "curated", tmp_path / "artifacts"
    cache_dir, texture = [artifacts / "experiments" / name for name in ("nested_texture_kernel", "physical_texture")]
    reports, submissions = artifacts / "reports", artifacts / "submissions"
    for directory in (curated, cache_dir, texture, reports, submissions):
        directory.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n  reports_dir: {reports}\n  submissions_dir: {submissions}\n"
        "files:\n  train: train.csv\n  sample_submission: sample.csv\n  ppm: ppm.csv\n"
    )
    pd.DataFrame({"phone": ["one", "two"], "ppm": [1., 1.]}).to_csv(curated / "ppm.csv", index=False)
    ids = ["F", "B", "E", "A", "C", "D"]
    curves = np.array([[v] * 10 + [100.] for v in (10., 20., 30., 70., 80., 90.)])
    truth = pd.DataFrame(curves, columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0, "sample_id", ids)
    truth.to_csv(curated / "train.csv", index=False)
    sample = truth.iloc[:2].copy()
    sample["sample_id"] = ["T2", "T1"]
    sample.to_csv(curated / "sample.csv", index=False)
    index = pd.DataFrame([
        {"split": "test" if soil.startswith("T") else "train", "sample_id": soil,
         "camera": camera, "path": str(tmp_path / f"{soil}_{camera}_{i}.jpg")}
        for soil in [*ids, "T1", "T2"] for i, camera in enumerate(("one", "one", "two"))
    ])
    for path in index["path"]:
        Path(path).write_bytes(b"photo fixture")
    index.to_csv(reports / "photo_index.csv", index=False)
    columns = [f"feature_{i}" for i in range(17)]
    photos = pd.concat([index, pd.DataFrame(np.random.default_rng(92).normal(size=(len(index), 17)), columns=columns)], axis=1)
    cache = cache_dir / "photo_features.csv"
    photos.iloc[::-1].to_csv(cache, index=False)
    oof, views = evaluate_samples(photos[photos["split"] == "train"], ids, curves, ridge_curves)
    reference = pd.DataFrame(oof, columns=CANONICAL_GRAIN_LABELS)
    reference.insert(0, "sample_id", ids)
    reference["emd"] = 999.
    for frame, name in ((reference, "oof"), (views, "camera")):
        frame.insert(0, "experiment", "ridge_rgb_texture_100")
        frame.iloc[::-1].to_csv(texture / f"{name}_predictions.csv", index=False)
    train = photos[photos["split"] == "train"].groupby("sample_id")[columns].mean().loc[ids].to_numpy()
    test = photos[photos["split"] == "test"].groupby("sample_id")[columns].mean().loc[sample["sample_id"]].to_numpy()
    sample[list(CANONICAL_GRAIN_LABELS)] = ridge_curves(train, curves, test)
    sample.iloc[::-1].to_csv(submissions / "ridge_rgb_texture_100.csv", index=False)
    (cache_dir / "summary.json").write_text(json.dumps({
        "sources": [{"path": str(curated / "train.csv"), "sha256": sha256((curated / "train.csv").read_bytes()).hexdigest()}],
        "feature_cache": {"path": str(cache), "sha256": sha256(cache.read_bytes()).hexdigest()},
    }))
    source_paths = [curated / "ppm.csv", *(Path(p) for p in index["path"])]
    (texture / "summary.json").write_text(json.dumps({
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in source_paths],
    }))
    rng = np.random.default_rng(11)
    spectrum = {p: row for p, row in zip(index["path"], rng.dirichlet(np.ones(6), len(index)), strict=True)}
    monkeypatch.setattr('soilgrain.physical_photo_experiment.physical_spectrum_features', lambda path, camera: spectrum[path])
    return config, artifacts, photos, spectrum



def test_physical_photo_run_reconstructs_reference_preserves_inputs_and_excludes_test(physical_photo_run):
    config, artifacts, photos, spectrum = physical_photo_run
    before = {p: p.read_bytes() for p in config.parent.rglob('*') if p.is_file()}
    paths = write_physical_photo_experiment(config)
    summary = json.loads(paths['summary'].read_text())
    assert max(summary['reference_max_abs_differences'].values()) < 1e-10
    assert [row['feature_count'] for row in summary['experiments']] == [23, 17]
    assert all(p.read_bytes() == content for p, content in before.items())
    for name in ('spectral_ridge', 'photo_ridge'):
        assert pd.read_csv(paths[f'{name}_submission'])['sample_id'].tolist() == ['T2', 'T1']
    assert len(pd.read_csv(paths['oof'])) == 12
    assert len(pd.read_csv(paths['camera_predictions'])) == 24
    assert all(sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
               for item in [*summary['sources'], *summary['photo_sources'], *summary['submissions']])
    retained = {key: paths[key].read_bytes() for key in ('oof', 'camera_predictions', 'comparison')}
    for path in photos.loc[photos['split'] == 'test', 'path']:
        spectrum[path] = np.array([1., 0., 0., 0., 0., 0.])
    write_physical_photo_experiment(config)
    assert all(paths[key].read_bytes() == value for key, value in retained.items())


@pytest.mark.parametrize('corruption', ['photo', 'calibration', 'camera_reference', 'cache'])
def test_physical_photo_run_rejects_changed_sources_and_reference(physical_photo_run, corruption):
    config, artifacts, photos, _spectrum = physical_photo_run
    if corruption == 'photo':
        Path(photos.iloc[0]['path']).write_bytes(b'changed')
    elif corruption == 'calibration':
        (config.parent / 'curated' / 'ppm.csv').write_text('phone,ppm\none,2\ntwo,2\n')
    else:
        path = (artifacts / 'experiments' / 'physical_texture' / 'camera_predictions.csv' if corruption == 'camera_reference'
                else artifacts / 'experiments' / 'nested_texture_kernel' / 'photo_features.csv')
        frame = pd.read_csv(path)
        frame.loc[0, '0.002' if corruption == 'camera_reference' else 'feature_0'] -= 0.01
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='reference|source changed'):
        write_physical_photo_experiment(config)
    assert not (artifacts / 'experiments' / 'physical_photo').exists()


@pytest.mark.parametrize('spectral,photo,expected', [
    ((42., 20.), (41., 25.), 'photo_ridge'),
    ((41., 20.), (41., 25.), 'spectral_ridge'),
    ((43., 20.), (42., 29.), None),
    ((42., None), (42., 30.), None),
    ((42., 20.), (40., 30.), 'spectral_ridge'),
])
def test_submission_choice_requires_strict_gain_in_both_metrics(spectral, photo, expected):
    rows = [{'experiment': name, 'loo_emd': error, 'paired_camera_disagreement_emd': camera}
            for name, (error, camera) in [('spectral_ridge', spectral), ('photo_ridge', photo)]]
    assert _select_candidate(rows, 43., 29.) == expected
