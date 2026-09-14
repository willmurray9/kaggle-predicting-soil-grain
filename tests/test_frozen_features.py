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


@pytest.mark.parametrize("preprocessing,size", [("legacy", 256), ("official", 224)])
def test_encoder_is_frozen_and_batch_independent_including_test_photos(tmp_path: Path, encoder_checkpoint, preprocessing, size) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    model, observations, _checkpoint = encoder_checkpoint
    index, ppm = _photos(tmp_path, ["red"] + ["white"] * 8)
    alone, _ = extract_frozen_features(index.iloc[:1], ppm, preprocessing=preprocessing)
    mixed, _ = extract_frozen_features(index, ppm, preprocessing=preprocessing)

    np.testing.assert_array_equal(alone.iloc[0, 4:], mixed.iloc[0, 4:])
    np.testing.assert_array_equal(model.bn.running_mean.numpy(), [0, 0, 0])
    np.testing.assert_array_equal(model.bn.running_var.numpy(), [1, 1, 1])
    assert not model.training
    assert all(not parameter.requires_grad and parameter.device.type == "cpu" for parameter in model.parameters())
    assert observations == [
        (False, False, True, (1, 3, size, size)),
        (False, False, True, (8, 3, size, size)),
        (False, False, True, (1, 3, size, size)),
    ]


def test_official_preprocessing_matches_recipe_on_center_and_border_pixels(tmp_path: Path, encoder_checkpoint) -> None:
    import torch
    from torchvision.models import ResNet18_Weights

    from soilgrain.frozen_features import extract_frozen_features

    index, ppm = _photos(tmp_path, ["red"])
    pixels = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
    pixels[16:-16, 16:-16] = np.random.default_rng(19).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    image.save(index.iloc[0]["path"])
    ppm.loc[0, ["width", "height", "ppm"]] = [256, 256, 2.56]
    encoder_inputs = []
    hook = encoder_checkpoint[0].register_forward_pre_hook(lambda _model, args: encoder_inputs.append(args[0].clone()))
    try:
        features, metadata = extract_frozen_features(index, ppm, preprocessing="official")
    finally:
        hook.remove()

    expected = ResNet18_Weights.IMAGENET1K_V1.transforms()(image)
    assert encoder_inputs[0].shape == (1, 3, 224, 224)
    torch.testing.assert_close(encoder_inputs[0][0], expected, rtol=0, atol=0)
    np.testing.assert_allclose(features.iloc[0, 4:].to_numpy(dtype=float), expected.mean(dim=(1, 2)).repeat(171)[:512].numpy(), atol=1e-6)
    assert metadata["crop"] == {
        "millimeters": 100, "pixels": [256, 256], "resample": "LANCZOS", "center_crop_pixels": [224, 224],
    }
    assert metadata["preprocessing"] == {
        "name": "official", "resize_size": [256], "interpolation": "BILINEAR", "antialias": True,
        "input_pixels": [224, 224], "nominal_field_mm": 87.5,
    }


def test_explicit_legacy_preserves_default_features_and_metadata(tmp_path: Path, encoder_checkpoint) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    index, ppm = _photos(tmp_path, ["red", "black"])
    default_features, default_metadata = extract_frozen_features(index, ppm)
    legacy_features, legacy_metadata = extract_frozen_features(index, ppm, preprocessing="legacy")

    pd.testing.assert_frame_equal(legacy_features, default_features)
    assert legacy_metadata == default_metadata
    assert "preprocessing" not in default_metadata
    assert default_metadata["crop"] == {
        "millimeters": 100, "pixels": [256, 256], "resample": "LANCZOS", "center_crop_pixels": None,
    }


def test_frozen_features_reject_unknown_preprocessing() -> None:
    from soilgrain.frozen_features import extract_frozen_features

    with pytest.raises(ValueError, match="preprocessing"):
        extract_frozen_features(pd.DataFrame(), pd.DataFrame(), preprocessing="unknown")


def test_frozen_features_reject_checkpoint_hash_mismatch(tmp_path: Path, encoder_checkpoint) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    encoder_checkpoint[2].write_bytes(b"changed cached weights")
    index, ppm = _photos(tmp_path, ["red"])
    with pytest.raises(ValueError, match="checkpoint.*hash"):
        extract_frozen_features(index, ppm)


@pytest.fixture
def mobilenet_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    observations = []

    class TinyMobileNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3, eps=0)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(960, 1280), torch.nn.Hardswish(),
                torch.nn.Dropout(), torch.nn.Linear(1280, 1000),
            )

        def forward(self, batch):
            observations.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(), tuple(batch.shape)))
            channels = self.bn(batch).mean(dim=(2, 3))
            return self.classifier(channels.repeat(1, 320))

    checkpoint_bytes = b"unit-test mobile checkpoint; no pretrained download"
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    filename = f"mobilenet_v3_large-{digest[:8]}.pth"
    checkpoint = tmp_path / "checkpoints" / filename
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(checkpoint_bytes)
    weights = SimpleNamespace(
        url=f"https://download.pytorch.org/models/{filename}",
        transforms=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms,
    )
    model = TinyMobileNet()

    def mobilenet_v3_large(*, weights: object):
        assert weights is torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        return model

    monkeypatch.setattr(torchvision.models, "MobileNet_V3_Large_Weights", SimpleNamespace(IMAGENET1K_V1=weights))
    monkeypatch.setattr(torchvision.models, "mobilenet_v3_large", mobilenet_v3_large)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    return model, observations, checkpoint


def test_mobilenet_uses_official_recipe_and_removes_entire_classifier(tmp_path: Path, mobilenet_checkpoint) -> None:
    import torch
    from torchvision.models import MobileNet_V3_Large_Weights

    from soilgrain.frozen_features import extract_frozen_features

    index, ppm = _photos(tmp_path, ["red"])
    pixels = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
    pixels[16:-16, 16:-16] = np.random.default_rng(19).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    image.save(index.iloc[0]["path"])
    ppm.loc[0, ["width", "height", "ppm"]] = [256, 256, 2.56]
    encoder_inputs = []
    hook = mobilenet_checkpoint[0].register_forward_pre_hook(lambda _model, args: encoder_inputs.append(args[0].clone()))
    try:
        features, metadata = extract_frozen_features(index, ppm, encoder_name="mobilenet_v3_large", preprocessing="official")
    finally:
        hook.remove()

    expected = MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()(image)
    torch.testing.assert_close(encoder_inputs[0][0], expected, rtol=0, atol=0)
    assert isinstance(mobilenet_checkpoint[0].classifier, torch.nn.Identity)
    assert features.shape == (1, 964)
    assert features.columns[4:].tolist() == [f"feature_{i}" for i in range(960)]
    pd.testing.assert_frame_equal(features.iloc[:, :4], index.reset_index(drop=True))
    np.testing.assert_allclose(features.iloc[0, 4:].to_numpy(dtype=float), expected.mean(dim=(1, 2)).repeat(320).numpy(), atol=1e-6)
    assert metadata["encoder"] == "mobilenet_v3_large"
    assert metadata["weights"] == "MobileNet_V3_Large_Weights.IMAGENET1K_V1"
    assert metadata["feature_count"] == 960
    assert metadata["checkpoint_sha256"] == hashlib.sha256(mobilenet_checkpoint[2].read_bytes()).hexdigest()
    assert metadata["preprocessing"] == {
        "name": "official", "resize_size": [256], "interpolation": "BILINEAR", "antialias": True,
        "input_pixels": [224, 224], "nominal_field_mm": 87.5,
    }


def test_mobilenet_is_frozen_and_batch_independent(tmp_path: Path, mobilenet_checkpoint) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    model, observations, _checkpoint = mobilenet_checkpoint
    index, ppm = _photos(tmp_path, ["red"] + ["white"] * 8)
    alone, _ = extract_frozen_features(index.iloc[:1], ppm, encoder_name="mobilenet_v3_large", preprocessing="official")
    mixed, _ = extract_frozen_features(index, ppm, encoder_name="mobilenet_v3_large", preprocessing="official")

    np.testing.assert_array_equal(alone.iloc[0, 4:], mixed.iloc[0, 4:])
    np.testing.assert_array_equal(model.bn.running_mean.numpy(), [0, 0, 0])
    np.testing.assert_array_equal(model.bn.running_var.numpy(), [1, 1, 1])
    assert not model.training
    assert all(not parameter.requires_grad and parameter.device.type == "cpu" for parameter in model.parameters())
    assert observations == [
        (False, False, True, (1, 3, 224, 224)),
        (False, False, True, (8, 3, 224, 224)),
        (False, False, True, (1, 3, 224, 224)),
    ]


@pytest.mark.parametrize("encoder,preprocessing", [("unknown", "official"), ("mobilenet_v3_large", "legacy")])
def test_frozen_features_reject_invalid_encoder_combinations(encoder: str, preprocessing: str) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    with pytest.raises(ValueError, match="encoder|official"):
        extract_frozen_features(pd.DataFrame(), pd.DataFrame(), encoder_name=encoder, preprocessing=preprocessing)


def test_mobilenet_rejects_checkpoint_hash_mismatch(tmp_path: Path, mobilenet_checkpoint) -> None:
    from soilgrain.frozen_features import extract_frozen_features

    mobilenet_checkpoint[2].write_bytes(b"changed cached mobile weights")
    index, ppm = _photos(tmp_path, ["red"])
    with pytest.raises(ValueError, match="checkpoint.*hash"):
        extract_frozen_features(index, ppm, encoder_name="mobilenet_v3_large", preprocessing="official")
