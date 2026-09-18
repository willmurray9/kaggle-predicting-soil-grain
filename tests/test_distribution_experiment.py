import numpy as np
import pandas as pd


def test_native_feature_cache_preserves_photo_keys_and_validates_cached_rows(tmp_path, monkeypatch):
    from soilgrain.distribution_experiment import native_feature_cache
    from soilgrain import native_texture
    calls = []

    def extract(path, camera):
        calls.append(str(path))
        return np.arange(23, dtype=float)

    monkeypatch.setattr(native_texture, 'native_spectral_features', extract)
    monkeypatch.setattr(native_texture, 'physical_lbp_features', lambda path, camera: np.ones(40)/10)
    index = pd.DataFrame({'split': ['train', 'test'], 'sample_id': ['S', 'T'],
                          'camera': ['Phone', 'Phone'], 'path': ['first.png', 'second.png']})
    cameras = pd.DataFrame({'phone': ['Phone'], 'ppm': [4.6]}).set_index('phone')
    actual = native_feature_cache(index, cameras, tmp_path)
    assert calls == ['first.png', 'second.png']
    for name, count in [('native_spectral', 23), ('lbp', 40)]:
        frame = actual[name]
        pd.testing.assert_frame_equal(frame[index.columns], index)
        assert frame.shape == (2, count+4)
    repeated = native_feature_cache(index, cameras, tmp_path)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(actual['native_spectral'], repeated['native_spectral'])
    import pytest
    with monkeypatch.context() as changed:
        changed.setattr(native_texture, 'SPECTRAL_BANDS_MM', ((0.5, 1.),) * 6)
        with pytest.raises(ValueError):
            native_feature_cache(index, cameras, tmp_path)
    # A cache cannot silently attach a different soil to a photo.
    path = tmp_path / 'native_spectral_photo_features.csv'
    wrong = pd.read_csv(path)
    wrong.loc[0, 'sample_id'] = 'WRONG'
    wrong.to_csv(path, index=False)
    with pytest.raises(ValueError):
        native_feature_cache(index, cameras, tmp_path)


def test_candidate_summary_aligns_reference_by_soil_and_records_actual_errors(tmp_path):
    from soilgrain.distribution_experiment import save_candidate
    from soilgrain.constants import CANONICAL_GRAIN_LABELS
    from soilgrain.ridge import ridge_curves
    rows=[]
    for i in range(4):
        for camera in ['Motorola Edge', 'Samsung A52']:
            rows.append({'sample_id': f'S{i}', 'split': 'train', 'camera': camera,
                         'feature_0': float(i), 'path': f'{i}_{camera}'})
    for sample_id, value in [('T2', 2.), ('T1', 1.)]:
        rows.append({'sample_id': sample_id, 'split': 'test', 'camera': 'iPhone',
                     'feature_0': value, 'path': sample_id})
    photos=pd.DataFrame(rows)
    truth=pd.DataFrame([[i*10]*10+[100] for i in range(4)], columns=CANONICAL_GRAIN_LABELS)
    truth.insert(0,'sample_id',[f'S{i}' for i in range(4)])
    sample=truth.iloc[:2].copy();sample['sample_id']=['T2','T1']
    reference=truth.copy()
    for i in range(4):
        reference.loc[i,list(CANONICAL_GRAIN_LABELS[:10])] += i+1
    reference=reference.iloc[::-1]
    result=save_candidate('synthetic',photos,ridge_curves,truth,sample,reference,tmp_path)
    oof=pd.read_csv(result['oof']['path'])
    np.testing.assert_allclose(oof.reference_emd,[5,10,15,20])
    assert len(pd.read_csv(result['camera_transfer']['path']))==16
    candidate=pd.read_csv(result['submission']['path'])
    assert candidate.sample_id.tolist()==['T2','T1']
    np.testing.assert_allclose(candidate.iloc[:,1:].to_numpy(),
                               [[16.42857142857143]*10+[100],[13.57142857142857]*10+[100]])
