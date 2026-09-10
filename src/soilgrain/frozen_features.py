from __future__ import annotations

from hashlib import file_digest
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import PIL

from soilgrain.image_model import physical_crop


def extract_frozen_features(index: pd.DataFrame, ppm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Extract fixed ImageNet ResNet-18 features from each calibrated photo crop."""
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    encoder = resnet18(weights=weights).to("cpu")
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).eval()
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    with checkpoint.open("rb") as source:
        checkpoint_hash = file_digest(source, "sha256").hexdigest()
    if not checkpoint_hash.startswith(checkpoint.stem.rsplit("-", 1)[1]):
        raise ValueError(f"Pretrained checkpoint has an invalid hash: {checkpoint}.")

    recipe = weights.transforms()
    mean = torch.tensor(recipe.mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(recipe.std, dtype=torch.float32)[:, None, None]
    cameras = ppm.set_index("phone")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(index), 8):
            images = []
            for row in index.iloc[start:start + 8].itertuples(index=False):
                crop = physical_crop(row.path, cameras.loc[row.camera])
                rgb = torch.from_numpy(np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)) / 255.0
                images.append((rgb - mean) / std)
            batches.append(encoder(torch.stack(images)).numpy())
    values = np.vstack(batches)
    if values.shape != (len(index), 512) or not np.isfinite(values).all():
        raise ValueError("The frozen encoder must produce 512 finite features per photo.")
    features = pd.concat([
        index[["split", "sample_id", "camera", "path"]].reset_index(drop=True),
        pd.DataFrame(values, columns=[f"feature_{i}" for i in range(512)]),
    ], axis=1)
    return features, {
        "encoder": "resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "checkpoint_url": weights.url,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_path": str(checkpoint),
        "versions": {
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__,
        },
        "device": "cpu", "batch_size": 8, "feature_count": 512,
        "crop": {"millimeters": 100, "pixels": [256, 256], "resample": "LANCZOS", "center_crop_pixels": None},
        "normalization": {"mean": recipe.mean, "std": recipe.std},
    }
