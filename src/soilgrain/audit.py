from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from soilgrain.config import load_config
from soilgrain.constants import CANONICAL_GRAIN_LABELS, SUPPORT_DIAMETERS
from soilgrain.data import load_working_tables
from soilgrain.io import ensure_dir, write_csv
from soilgrain.metrics import emd_score
from soilgrain.targets import ordered_grain_columns


SAMPLE_IDS = ("F827", "H038", "H374")
KNN_EXPERIMENTS = (
    "reference_rgb_100", "gray_100", "normalized_gray_100", "gray_50", "gray_150"
)
PLOT_LABELS = {
    "reference_rgb_100": "RGB 100 mm",
    "ridge_rgb_100": "Ridge RGB 100 mm",
    "gray_100": "Gray 100 mm",
    "normalized_gray_100": "Normalized gray 100 mm",
    "gray_50": "Gray 50 mm",
    "gray_150": "Gray 150 mm",
}


def _physical_crop(path: str, camera: pd.Series, crop_mm: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    ppm = float(camera["ppm"]) * max(image.size) / max(camera["width"], camera["height"])
    side = round(crop_mm * ppm)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _neighbor_table(first_batch: Path, train_ids: list[str]) -> pd.DataFrame:
    rows = []
    for experiment in KNN_EXPERIMENTS:
        photos = pd.read_csv(first_batch / f"{experiment}_photo_features.csv")
        columns = [c for c in photos if c.startswith("feature_")]
        features = photos.groupby("sample_id")[columns].mean().loc[train_ids]
        for sample_id in SAMPLE_IDS:
            train = features.drop(index=sample_id)
            scale = train.to_numpy().std(axis=0)
            scale[scale < 1e-12] = 1.0
            distances = np.square((train.to_numpy() - features.loc[sample_id].to_numpy()) / scale).sum(axis=1)
            order = np.argsort(distances, kind="stable")[:3]
            for rank, position in enumerate(order, 1):
                rows.append({
                    "sample_id": sample_id, "experiment": experiment, "rank": rank,
                    "neighbor": train.index[position], "squared_standardized_distance": distances[position],
                })
    return pd.DataFrame(rows)


def _plot_sample(
    sample_id: str, photos: pd.DataFrame, cameras: pd.DataFrame, train: pd.DataFrame,
    oof: pd.DataFrame, neighbors: pd.DataFrame, output: Path,
) -> None:
    fig = plt.figure(figsize=(14, 2.15 * len(photos) + 5.5), layout="constrained")
    grid = fig.add_gridspec(len(photos) + 2, 4, height_ratios=[1] * len(photos) + [1.1, 1.1])
    for row_number, row in enumerate(photos.itertuples(index=False)):
        camera = cameras.loc[row.camera]
        with Image.open(row.path) as source:
            full = ImageOps.exif_transpose(source).convert("RGB")
        images = [full] + [_physical_crop(row.path, camera, size) for size in (50, 100, 150)]
        for column, image in enumerate(images):
            ax = fig.add_subplot(grid[row_number, column])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_number == 0:
                ax.set_title(("Full frame", "50 mm center", "100 mm center", "150 mm center")[column])
            if column == 0:
                ax.set_ylabel(f"{row.camera}\n{Path(row.path).name}", fontsize=8)

    curve_ax = fig.add_subplot(grid[-2:, :3])
    true = train.set_index("sample_id").loc[sample_id, ordered_grain_columns(train)].to_numpy(float)
    curve_ax.plot(SUPPORT_DIAMETERS, true, color="black", linewidth=3, label="Measured curve")
    for _, row in oof[oof["sample_id"] == sample_id].iterrows():
        values = row[list(CANONICAL_GRAIN_LABELS)].to_numpy(float)
        curve_ax.plot(SUPPORT_DIAMETERS, values, linewidth=1.5,
                      label=f"{PLOT_LABELS[row.experiment]} ({row.emd:.1f} EMD)")
    curve_ax.set(xscale="log", xlabel="Grain diameter (mm)", ylabel="Cumulative mass finer (%)", ylim=(0, 102))
    curve_ax.set_xticks(SUPPORT_DIAMETERS, CANONICAL_GRAIN_LABELS, rotation=35, ha="right")
    curve_ax.grid(True, which="both", alpha=0.2)
    curve_ax.legend(fontsize=8, ncols=2)

    text_ax = fig.add_subplot(grid[-2:, 3])
    text_ax.axis("off")
    lines = ["Held-out 3-NN neighbors"]
    selected = neighbors[neighbors["sample_id"] == sample_id]
    for experiment in KNN_EXPERIMENTS:
        names = selected[selected["experiment"] == experiment]["neighbor"].tolist()
        lines.append(f"{PLOT_LABELS[experiment]}:\n  {', '.join(names)}")
    text_ax.text(0, 1, "\n".join(lines), va="top", fontsize=9, linespacing=1.35)
    fig.suptitle(f"{sample_id}: all frames, calibrated center crops, and held-out predictions", fontsize=15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_audit(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train, _test, _sample, ppm = load_working_tables(cfg)
    first_batch = cfg.artifacts_dir / "experiments" / "first_batch"
    output = ensure_dir(cfg.reports_dir / "error_audit")
    photos = pd.read_csv(cfg.reports_dir / "photo_index.csv")
    oof = pd.read_csv(first_batch / "oof_predictions.csv")
    camera_predictions = pd.read_csv(first_batch / "camera_predictions.csv")
    train_ids = train["sample_id"].astype(str).tolist()
    neighbors = _neighbor_table(first_batch, train_ids)
    paired = []
    for (experiment, sample_id), group in camera_predictions.groupby(["experiment", "sample_id"]):
        curves = group[list(CANONICAL_GRAIN_LABELS)].to_numpy()
        paired.append({"experiment": experiment, "sample_id": sample_id,
                       "camera_disagreement_emd": emd_score(curves[0], curves[1]) if len(curves) == 2 else np.nan})
    metrics = camera_predictions.merge(oof[["experiment", "sample_id", "emd"]],
                                       on=["experiment", "sample_id"], suffixes=("_camera", "_pooled"))
    metrics = metrics.merge(pd.DataFrame(paired), on=["experiment", "sample_id"])
    paths = {
        "neighbors": write_csv(neighbors, output / "neighbors.csv"),
        "metrics": write_csv(metrics.query("sample_id in @SAMPLE_IDS")
                             [["sample_id", "experiment", "camera", "emd_pooled", "emd_camera",
                               "camera_disagreement_emd"]], output / "metrics.csv"),
    }
    cameras = ppm.set_index("phone")
    for sample_id in SAMPLE_IDS:
        path = output / f"{sample_id}.png"
        _plot_sample(sample_id, photos[photos["sample_id"] == sample_id], cameras, train, oof, neighbors, path)
        paths[sample_id] = path
    return paths


if __name__ == "__main__":
    for name, path in write_audit().items():
        print(f"{name}: {path}")
