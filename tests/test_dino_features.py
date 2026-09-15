import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain import dino_features


@pytest.fixture
def dino_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    observations = []
    inputs = []

    class TinyDino(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3, eps=0)

        def forward(self, _batch):
            raise AssertionError("Use patch tokens, not the class-token forward method.")

        def forward_features(self, batch):
            observations.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(), tuple(batch.shape)))
            inputs.append(batch.clone())
            channels = self.bn(batch).mean(dim=(2, 3)).repeat(1, 128)
            patches = channels[:, None, :] + torch.arange(256, dtype=torch.float32)[None, :, None] / 128
            return {"x_norm_patchtokens": patches, "x_norm_clstoken": torch.full_like(channels, 1e6)}

    model = TinyDino()
    checkpoint = tmp_path / "checkpoints" / "dinov2_vits14_pretrain.pth"
    checkpoint.parent.mkdir()
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setattr(dino_features, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    source = tmp_path / f"facebookresearch_dinov2_{dino_features.SOURCE_COMMIT}"
    source.mkdir()
    (source / "hubconf.py").write_text("# synthetic pinned source\n")
    source_hash = hashlib.sha256(
        f'hubconf.py\0{hashlib.sha256((source / "hubconf.py").read_bytes()).hexdigest()}\n'.encode()
    ).hexdigest()
    monkeypatch.setattr(dino_features, "SOURCE_SHA256", source_hash)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))

    def load(repo, entrypoint, **kwargs):
        assert repo == f"facebookresearch/dinov2:{dino_features.SOURCE_COMMIT}"
        assert entrypoint == "dinov2_vits14"
        assert kwargs == {"pretrained": False, "trust_repo": True, "verbose": False}
        return model

    def no_download(*_args, **_kwargs):
        raise AssertionError("Unit tests must not download models or code.")

    monkeypatch.setattr(torch.hub, "load", load)
    monkeypatch.setattr(torch.hub, "download_url_to_file", no_download)
    return model, observations, inputs, checkpoint, source


def _photos(tmp_path, colors):
    rows = []
    for i, color in enumerate(colors):
        path = tmp_path / f"photo_{i}.png"
        Image.new("RGB", (100, 100), color).save(path)
        rows.append({"split": "train" if i % 2 else "test", "sample_id": f"soil_{i}", "camera": "Phone", "path": str(path)})
    index = pd.DataFrame(rows, index=np.arange(len(rows))[::-1] + 20)
    ppm = pd.DataFrame({"phone": ["Phone"], "width": [100], "height": [100], "ppm": [1.0]})
    return index, ppm


def test_patch_mean_preserves_alignment_and_magnitude_without_class_token(tmp_path, dino_checkpoint):
    index, ppm = _photos(tmp_path, ["red", "black"])

    features, metadata = dino_features.extract_dino_features(index, ppm)

    pd.testing.assert_frame_equal(features.iloc[:, :4], index.reset_index(drop=True))
    assert features.shape == (2, 388)
    assert features.columns[4:].tolist() == [f"feature_{i}" for i in range(384)]
    assert features.iloc[:, 4:].to_numpy().dtype == np.float32
    # Every channel receives the mean token offset 0..255 divided by 128.
    expected = np.tile(np.array([2.2489083, -2.0357143, -1.8044444]) + 255 / 256, 128)
    np.testing.assert_allclose(features.iloc[0, 4:].to_numpy(dtype=float), expected, atol=1e-5)
    assert not np.isclose(np.linalg.norm(features.iloc[0, 4:].to_numpy(dtype=float)), 1.0)
    assert metadata["checkpoint_sha256"] == hashlib.sha256(dino_checkpoint[3].read_bytes()).hexdigest()
    assert metadata["source_commit"] == dino_features.SOURCE_COMMIT
    assert metadata["source_sha256"] == dino_features.SOURCE_SHA256
    assert metadata["source_files"] == [{
        "path": str(dino_checkpoint[4] / "hubconf.py"),
        "sha256": hashlib.sha256((dino_checkpoint[4] / "hubconf.py").read_bytes()).hexdigest(),
    }]
    assert metadata["pooling"] == "mean of last-layer learned-LayerNorm patch tokens; no class token or L2 normalization"


def test_recipe_uses_center_224_pixels_and_imagenet_normalization(tmp_path, dino_checkpoint):
    import torch

    index, ppm = _photos(tmp_path, ["red"])
    pixels = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
    pixels[16:-16, 16:-16] = np.random.default_rng(19).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(index.iloc[0]["path"])
    ppm.loc[0, ["width", "height", "ppm"]] = [256, 256, 2.56]

    _, metadata = dino_features.extract_dino_features(index, ppm)

    rgb = torch.tensor(pixels[16:-16, 16:-16].transpose(2, 0, 1), dtype=torch.float32) / 255
    expected = (rgb - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    torch.testing.assert_close(dino_checkpoint[2][0][0], expected, rtol=0, atol=0)
    assert metadata["preprocessing"] == {
        "name": "official", "resize_size": [256], "interpolation": "BICUBIC", "antialias": True,
        "input_pixels": [224, 224], "nominal_field_mm": 87.5,
    }
    assert metadata["crop"] == {
        "millimeters": 100, "pixels": [256, 256], "resample": "LANCZOS", "center_crop_pixels": [224, 224],
    }


def test_encoder_is_frozen_and_batch_independent(tmp_path, dino_checkpoint):
    model, observations, _inputs, _checkpoint, _source = dino_checkpoint
    index, ppm = _photos(tmp_path, ["red"] + ["white"] * 8)

    alone, _ = dino_features.extract_dino_features(index.iloc[:1], ppm)
    mixed, _ = dino_features.extract_dino_features(index, ppm)

    np.testing.assert_array_equal(alone.iloc[0, 4:], mixed.iloc[0, 4:])
    np.testing.assert_array_equal(model.bn.running_mean.numpy(), [0, 0, 0])
    np.testing.assert_array_equal(model.bn.running_var.numpy(), [1, 1, 1])
    assert all(not parameter.requires_grad and parameter.device.type == "cpu" for parameter in model.parameters())
    assert observations == [
        (False, False, True, (1, 3, 224, 224)),
        (False, False, True, (8, 3, 224, 224)),
        (False, False, True, (1, 3, 224, 224)),
    ]


@pytest.mark.parametrize("corrupted", ["checkpoint", "source"])
def test_corrupt_cached_bytes_are_rejected_before_loading_weights(tmp_path, dino_checkpoint, monkeypatch, corrupted):
    import torch

    index, ppm = _photos(tmp_path, ["red"])
    path = dino_checkpoint[3] if corrupted == "checkpoint" else dino_checkpoint[4] / "hubconf.py"
    path.write_bytes(b"changed cached bytes")

    def no_load(*_args, **_kwargs):
        raise AssertionError("Do not deserialize checkpoint until integrity checks pass.")

    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(ValueError, match=f"{corrupted}.*hash"):
        dino_features.extract_dino_features(index, ppm)


@pytest.mark.parametrize("problem", ["token_count", "feature_count", "nonfinite", "overflow"])
def test_encoder_rejects_invalid_patch_outputs(tmp_path, dino_checkpoint, monkeypatch, problem):
    import torch

    index, ppm = _photos(tmp_path, ["red"])

    def invalid_features(batch):
        patches = torch.zeros(len(batch), 255 if problem == "token_count" else 256, 383 if problem == "feature_count" else 384)
        if problem == "nonfinite":
            patches[0, 0, 0] = float("nan")
        elif problem == "overflow":
            patches.fill_(3e38)
        return {"x_norm_patchtokens": patches}

    monkeypatch.setattr(dino_checkpoint[0], "forward_features", invalid_features)
    with pytest.raises(ValueError, match="patch tokens"):
        dino_features.extract_dino_features(index, ppm)
