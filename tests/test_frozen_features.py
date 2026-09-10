import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soilgrain import image_model


@pytest.mark.parametrize("crop_mm,box", [(100, (150, 100, 250, 200)), (50, (175, 125, 225, 175))])
def test_physical_crop_preserves_calibrated_pixels(tmp_path: Path, crop_mm: int, box: tuple) -> None:
    pixels = np.random.default_rng(7).integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    path = tmp_path / "downsampled.png"
    image.save(path)
    camera = pd.Series({"ppm": 2.0, "width": 800, "height": 600})

    actual = image_model.physical_crop(path, camera, crop_mm=crop_mm)
    expected = image.crop(box).resize((256, 256), Image.Resampling.LANCZOS)

    assert actual.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    if crop_mm == 100:
        np.testing.assert_array_equal(np.asarray(image_model.physical_crop(path, camera)), np.asarray(expected))


@pytest.fixture
def encoder_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    observations = []

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3, eps=0)
            self.fc = torch.nn.Linear(512, 1000)

        def forward(self, batch):
            observations.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(), tuple(batch.shape)))
            channels = self.bn(batch).mean(dim=(2, 3))
            return self.fc(channels.repeat(1, 171)[:, :512])

    checkpoint_bytes = b"unit-test checkpoint; no pretrained download"
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    filename = f"resnet18-{digest[:8]}.pth"
    checkpoint = tmp_path / "checkpoints" / filename
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(checkpoint_bytes)
    weights = SimpleNamespace(
        url=f"https://download.pytorch.org/models/{filename}",
        transforms=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms,
    )
    model = TinyEncoder()

    def resnet18(*, weights: object):
        assert weights is torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        return model

    monkeypatch.setattr(torchvision.models, "ResNet18_Weights", SimpleNamespace(IMAGENET1K_V1=weights))
    monkeypatch.setattr(torchvision.models, "resnet18", resnet18)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    return model, observations, checkpoint


def _photos(tmp_path: Path, colors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, color in enumerate(colors):
        path = tmp_path / f"photo_{i}.png"
        Image.new("RGB", (100, 100), color).save(path)
        rows.append({"split": "train" if i % 2 else "test", "sample_id": f"soil_{i}", "camera": "Phone", "path": str(path)})
    index = pd.DataFrame(rows, index=np.arange(len(rows))[::-1] + 20)
    ppm = pd.DataFrame({"phone": ["Phone"], "width": [100], "height": [100], "ppm": [1.0]})
    return index, ppm


def test_frozen_features_preserve_photo_alignment_and_imagenet_normalization(tmp_path: Path, encoder_checkpoint) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    index, ppm = _photos(tmp_path, ["red", "black"])
    features, metadata = extract_frozen_features(index, ppm)

    pd.testing.assert_frame_equal(features.iloc[:, :4].reset_index(drop=True), index.reset_index(drop=True))
    assert features.shape == (2, 516)
    assert features.columns[4:].tolist() == [f"feature_{i}" for i in range(512)]
    assert np.isfinite(features.iloc[:, 4:].to_numpy()).all()
    np.testing.assert_allclose(features.iloc[0, 4:7].to_numpy(dtype=float), [2.2489083, -2.0357143, -1.8044444], atol=1e-5)
    np.testing.assert_allclose(features.iloc[1, 4:7].to_numpy(dtype=float), [-2.1179039, -2.0357143, -1.8044444], atol=1e-5)
    assert metadata["checkpoint_sha256"] == hashlib.sha256(encoder_checkpoint[2].read_bytes()).hexdigest()
    assert metadata["checkpoint_url"].endswith(encoder_checkpoint[2].name)
    assert metadata["normalization"] == {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    assert metadata["versions"]["torch"]
    assert metadata["versions"]["torchvision"]


def test_encoder_is_frozen_and_batch_independent_including_test_photos(tmp_path: Path, encoder_checkpoint) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    model, observations, _checkpoint = encoder_checkpoint
    index, ppm = _photos(tmp_path, ["red"] + ["white"] * 8)
    alone, _ = extract_frozen_features(index.iloc[:1], ppm)
    mixed, _ = extract_frozen_features(index, ppm)

    np.testing.assert_array_equal(alone.iloc[0, 4:], mixed.iloc[0, 4:])
    np.testing.assert_array_equal(model.bn.running_mean.numpy(), [0, 0, 0])
    np.testing.assert_array_equal(model.bn.running_var.numpy(), [1, 1, 1])
    assert not model.training
    assert all(not parameter.requires_grad and parameter.device.type == "cpu" for parameter in model.parameters())
    assert observations == [
        (False, False, True, (1, 3, 256, 256)),
        (False, False, True, (8, 3, 256, 256)),
        (False, False, True, (1, 3, 256, 256)),
    ]


def test_frozen_features_reject_checkpoint_hash_mismatch(tmp_path: Path, encoder_checkpoint) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    encoder_checkpoint[2].write_bytes(b"changed cached weights")
    index, ppm = _photos(tmp_path, ["red"])
    with pytest.raises(ValueError, match="checkpoint.*hash"):
        extract_frozen_features(index, ppm)
