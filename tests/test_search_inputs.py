import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.search_inputs import _verified_sources, load_search_inputs


def record(path):
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("field", ["feature_cache", "submission", "checkpoint", "source_files"])
def test_source_tree_checks_nested_records(tmp_path, field):
    leaf = tmp_path / "leaf.bin"
    leaf.write_bytes(b"original")
    item = record(leaf)
    content = ({"encoder": {"checkpoint_path": str(leaf), "checkpoint_sha256": item["sha256"]}}
               if field == "checkpoint" else {field: [item] if field == "source_files" else item})
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps(content))
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"sources": [record(nested)]}))
    pinned = [record(parent)]
    verified = _verified_sources(pinned)
    assert {Path(item["path"]) for item in verified} == {leaf, nested, parent}
    leaf.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        _verified_sources(pinned)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    curated, artifacts = tmp_path / "curated", tmp_path / "artifacts"
    reports, submissions = artifacts / "reports", artifacts / "submissions"
    for folder in (curated, reports, submissions):
        folder.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  curated_dir: {curated}\n  artifacts_dir: {artifacts}\n  reports_dir: {reports}\n  submissions_dir: {submissions}\nfiles:\n  train: train.csv\n  sample_submission: sample.csv\n  ppm: ppm.csv\n")

    def curves(ids):
        frame = pd.DataFrame(np.tile(np.linspace(0, 100, 11), (len(ids), 1)), columns=CANONICAL_GRAIN_LABELS)
        frame.insert(0, "sample_id", ids)
        return frame

    curves(["B", "A", "C"]).to_csv(curated / "train.csv", index=False)
    curves(["T2", "T1"]).to_csv(curated / "sample.csv", index=False)
    pd.DataFrame({"phone": ["one", "two"], "ppm": [1., 2.]}).to_csv(curated / "ppm.csv", index=False)
    rows = []
    for soil in ["A", "C", "B", "T1", "T2"]:
        for camera in ["two", "one"]:
            photo = tmp_path / f"{soil}_{camera}.jpg"
            photo.write_bytes(b"photo")
            rows.append({"split": "test" if soil.startswith("T") else "train", "sample_id": soil, "camera": camera, "path": str(photo)})
    index = pd.DataFrame(rows)
    index.to_csv(reports / "photo_index.csv", index=False)
    base_sources = [record(p) for p in [config, *curated.iterdir(), reports / "photo_index.csv"]]
    pins = {}
    manifests = []
    names = {"physical_photo": "rgb_texture_spectral_ridge.csv", "dino_pca": "dinov2_vits14_patch_pca8_nested_ridge.csv", "mobilenet_pca": "mobilenet_v3_large_pca8_nested_ridge.csv"}
    for name in ["physical_photo", "dino_pca", "mobilenet_pca", "patch_mixture"]:
        folder = artifacts / "experiments" / name
        folder.mkdir(parents=True)
        manifest = {"sources": base_sources, "photo_sources": [record(Path(p)) for p in index.path]}
        if name in ("dino_pca", "patch_mixture"):
            count = 384 if name == "dino_pca" else 29
            features = pd.DataFrame(np.arange(len(index) * count).reshape(len(index), count) / 7., columns=[f"feature_{i}" for i in range(count)])
            cache = folder / "photo_features.csv"
            pd.concat([index, features], axis=1).iloc[::-1].to_csv(cache, index=False)
            manifest["feature_cache"] = record(cache)
        if name in names:
            for kind in ["oof", "camera"]:
                frame = curves(["C", "A", "B"] if kind == "oof" else ["C", "C", "B", "B", "A", "A"])
                if kind == "camera":
                    frame.insert(1, "camera", ["one", "two"] * 3)
                if name == "physical_photo":
                    frame.insert(0, "experiment", "spectral_ridge")
                path = folder / f"{kind}_predictions.csv"
                frame.to_csv(path, index=False)
                pins[str(path.relative_to(artifacts))] = record(path)["sha256"]
            candidate = submissions / names[name]
            curves(["T1", "T2"]).to_csv(candidate, index=False)
            manifest["submission"] = record(candidate)
        path = folder / "summary.json"
        path.write_text(json.dumps(manifest))
        manifests.append(path)
        pins[str(path.relative_to(artifacts))] = record(path)["sha256"]
    monkeypatch.setattr("soilgrain.search_inputs.PINNED_INPUTS", pins)
    return config, artifacts, index, manifests, pins


def test_search_inputs_aligns_without_mutating_and_preserves_feature_precision(inputs):
    config, artifacts, index, _manifests, _pins = inputs
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    truth, sample, photos, components, sources = load_search_inputs(config)
    assert truth.sample_id.tolist() == ["B", "A", "C"]
    assert photos["spectral"].shape == (10, 27)
    assert photos["dino"].shape == (10, 388)
    assert photos["spectral"].filter(like="feature_").dtypes.eq(np.float64).all()
    assert photos["dino"].filter(like="feature_").dtypes.eq(np.float32).all()
    pd.testing.assert_frame_equal(photos["spectral"][index.columns], index)
    for component in components.values():
        assert component["oof"].sample_id.tolist() == truth.sample_id.tolist()
        assert component["submission"].sample_id.tolist() == sample.sample_id.tolist()
        pd.testing.assert_frame_equal(component["camera"][["sample_id", "camera"]], index.loc[index.split == "train", ["sample_id", "camera"]].reset_index(drop=True))
    assert all(record(Path(item["path"])) == item for item in sources)
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("corruption", ["pinned_curve", "camera_coverage", "labels", "photo", "cache"])
def test_search_inputs_rejects_corrupt_or_misaligned_inputs(inputs, corruption):
    config, artifacts, index, manifests, pins = inputs
    if corruption in ("pinned_curve", "camera_coverage"):
        path = artifacts / "experiments/dino_pca/camera_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "camera"] = "wrong"
        frame.to_csv(path, index=False)
        if corruption == "camera_coverage":
            pins[str(path.relative_to(artifacts))] = record(path)["sha256"]
    elif corruption == "labels":
        path = config.parent / "curated/train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "sample_id"] = "T1"
        frame.to_csv(path, index=False)
    elif corruption == "photo":
        Path(index.iloc[0].path).write_bytes(b"changed photo")
    else:
        path = artifacts / "experiments/patch_mixture/photo_features.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "feature_1"] = np.nan
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source changed|coverage|disjoint"):
        load_search_inputs(config)
