from hashlib import file_digest, sha256
from pathlib import Path

import numpy as np
import pandas as pd
import PIL

from soilgrain.image_model import physical_crop


SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_SHA256 = "117c2d9278ba282500e5d3924dae634672bd19cf892f3f8f4800fbd80b7b6af4"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
CHECKPOINT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def _file_hash(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def extract_dino_features(index: pd.DataFrame, ppm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Average frozen DINOv2-S/14 patch tokens for each calibrated photo crop."""
    import torch
    import torchvision
    from torchvision import transforms

    if index.empty:
        raise ValueError("DINO feature extraction requires at least one photo.")
    cache = Path(torch.hub.get_dir())
    checkpoint = cache / "checkpoints" / "dinov2_vits14_pretrain.pth"
    if not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(checkpoint), hash_prefix=CHECKPOINT_SHA256, progress=False)
    checkpoint_hash = _file_hash(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(f"DINO checkpoint has an invalid hash: {checkpoint}.")

    encoder = torch.hub.load(
        f"facebookresearch/dinov2:{SOURCE_COMMIT}", "dinov2_vits14",
        pretrained=False, trust_repo=True, verbose=False,
    )
    source_path = cache / f"facebookresearch_dinov2_{SOURCE_COMMIT}"
    source_files = [
        {"path": str(path), "sha256": _file_hash(path)}
        for path in sorted([source_path / "hubconf.py", *(source_path / "dinov2").rglob("*.py")])
    ]
    source_digest = sha256("".join(
        f'{Path(record["path"]).relative_to(source_path).as_posix()}\0{record["sha256"]}\n'
        for record in source_files
    ).encode()).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise ValueError(f"DINO source has an invalid hash: {source_path}.")
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    encoder.to(device="cpu", dtype=torch.float32).requires_grad_(False).eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    recipe = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    cameras = ppm.set_index("phone")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(index), 8):
            images = [
                recipe(physical_crop(row.path, cameras.loc[row.camera]))
                for row in index.iloc[start:start + 8].itertuples(index=False)
            ]
            tokens = encoder.forward_features(torch.stack(images))["x_norm_patchtokens"]
            if tokens.shape != (len(images), 256, 384) or not torch.isfinite(tokens).all():
                raise ValueError("DINO must produce 256 finite patch tokens with 384 features per photo.")
            batches.append(tokens.mean(dim=1).numpy())
    values = np.vstack(batches).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("DINO patch tokens must yield finite pooled features.")
    features = pd.concat([
        index[["split", "sample_id", "camera", "path"]].reset_index(drop=True),
        pd.DataFrame(values, columns=[f"feature_{i}" for i in range(384)]),
    ], axis=1)
    metadata = {
        "encoder": "dinov2_vits14", "weights": "LVD142M",
        "checkpoint_url": CHECKPOINT_URL, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_path": str(checkpoint),
        "source_repository": "https://github.com/facebookresearch/dinov2",
        "source_commit": SOURCE_COMMIT, "source_path": str(source_path),
        "source_sha256": source_digest, "source_files": source_files,
        "versions": {
            "torch": str(torch.__version__), "torchvision": torchvision.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__,
        },
        "device": "cpu", "batch_size": 8, "torch_threads": torch.get_num_threads(),
        "feature_count": 384, "feature_dtype": "float32", "patch_size": 14, "patch_count": 256,
        "pooling": "mean of last-layer learned-LayerNorm patch tokens; no class token or L2 normalization",
        "crop": {"millimeters": 100, "pixels": [256, 256], "resample": "LANCZOS", "center_crop_pixels": [224, 224]},
        "normalization": {"mean": mean, "std": std},
        "preprocessing": {
            "name": "official", "resize_size": [256], "interpolation": "BICUBIC", "antialias": True,
            "input_pixels": [224, 224], "nominal_field_mm": 87.5,
        },
    }
    return features, metadata
