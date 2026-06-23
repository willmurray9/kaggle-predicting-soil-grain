from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from soilgrain.baselines import equal_bin_curve, leave_one_out_score
from soilgrain.config import load_config
from soilgrain.data import _find_photo_dir, index_photos, load_working_tables
from soilgrain.io import ensure_dir, write_json
from soilgrain.metrics import emd_score
from soilgrain.targets import cumulative_to_bin_masses, curve_array, ordered_grain_columns, validate_cumulative_curves


def _plot_label_curves(train: pd.DataFrame, out_path: Path) -> None:
    cols = ordered_grain_columns(train)
    x = np.array([float(c) for c in cols])
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in train.iterrows():
        ax.plot(x, row[cols].to_numpy(dtype=float), alpha=0.55, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Grain diameter (mm)")
    ax.set_ylabel("Cumulative mass finer (%)")
    ax.set_title("Training Grain Size Curves")
    ax.set_ylim(0, 102)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_bin_heatmap(train: pd.DataFrame, out_path: Path) -> None:
    masses = cumulative_to_bin_masses(train)
    fig, ax = plt.subplots(figsize=(10, max(5, len(masses) * 0.24)))
    image = ax.imshow(masses.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(masses.columns)))
    ax.set_xticklabels(masses.columns, rotation=45, ha="right")
    if "sample_id" in train.columns and len(train) <= 60:
        ax.set_yticks(range(len(train)))
        ax.set_yticklabels(train["sample_id"].astype(str))
    ax.set_title("Mass Share by Grain-Size Bin")
    fig.colorbar(image, ax=ax, label="Mass share (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _placeholder_contact_sheet(out_path: Path, message: str) -> None:
    canvas = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 110), message, fill="black")
    canvas.save(out_path)


def _write_contact_sheet(photo_index: pd.DataFrame, out_path: Path, max_images: int = 24) -> None:
    if photo_index.empty:
        _placeholder_contact_sheet(out_path, "No photos were indexed.")
        return

    readable: list[tuple[Image.Image, str]] = []
    for row in photo_index[photo_index["is_matched"]].head(max_images).itertuples(index=False):
        try:
            image = Image.open(row.path).convert("RGB")
            image.thumbnail((180, 140))
            readable.append((image.copy(), f"{row.sample_id}  {row.camera or ''}".strip()))
        except Exception:
            continue
    if not readable:
        _placeholder_contact_sheet(out_path, "No readable indexed images found.")
        return

    cols = 4
    tile_w, tile_h = 220, 185
    rows = int(np.ceil(len(readable) / cols))
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(readable):
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h
        canvas.paste(image, (x + 20, y + 10))
        draw.text((x + 20, y + 155), label[:30], fill="black")
    canvas.save(out_path)


def _pairwise_label_emd(train: pd.DataFrame) -> float | None:
    values = curve_array(train)
    if len(values) < 2:
        return None
    scores = [emd_score(values[[i]], values[[j]]) for i, j in combinations(range(len(values)), 2)]
    return float(np.mean(scores))


def _load_or_build_photo_index(cfg, train: pd.DataFrame, test: pd.DataFrame, ppm: pd.DataFrame) -> pd.DataFrame:
    index_path = cfg.reports_dir / "photo_index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)
    train_dir = _find_photo_dir(cfg.raw_dir, cfg.photo_dirs["training"])
    test_dir = _find_photo_dir(cfg.raw_dir, cfg.photo_dirs["test"])
    photo_index = index_photos(train_dir, test_dir, train["sample_id"], test["sample_id"], ppm)
    ensure_dir(cfg.reports_dir)
    photo_index.to_csv(index_path, index=False)
    return photo_index


def write_label_reports(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train, _test, _sample, _ppm = load_working_tables(cfg)
    validate_cumulative_curves(train)
    ensure_dir(cfg.reports_dir)
    paths = {
        "label_curves": cfg.reports_dir / "label_curves.png",
        "bin_mass_heatmap": cfg.reports_dir / "bin_mass_heatmap.png",
    }
    _plot_label_curves(train, paths["label_curves"])
    _plot_bin_heatmap(train, paths["bin_mass_heatmap"])
    return paths


def write_eda_reports(config_path: str | Path = "configs/data.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    train, test, _sample, ppm = load_working_tables(cfg)
    report_paths = write_label_reports(config_path)
    photo_index = _load_or_build_photo_index(cfg, train, test, ppm)

    contact_sheet = cfg.reports_dir / "photo_contact_sheet.png"
    _write_contact_sheet(photo_index, contact_sheet)

    train_values = curve_array(train)
    equal_score = emd_score(train_values, np.tile(equal_bin_curve(), (len(train), 1)))
    summary = {
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "train_photos": int((photo_index["split"] == "train").sum()) if not photo_index.empty else 0,
        "test_photos": int((photo_index["split"] == "test").sum()) if not photo_index.empty else 0,
        "unmatched_photos": int((photo_index["issue"] == "unmatched_sample_id").sum()) if not photo_index.empty else 0,
        "camera_count": int(photo_index["camera"].replace("", np.nan).nunique()) if not photo_index.empty else 0,
        "equal_bin_train_score": float(equal_score),
        "mean_leave_one_out_score": leave_one_out_score(train, "mean"),
        "median_leave_one_out_score": leave_one_out_score(train, "median"),
        "mean_pairwise_label_emd": _pairwise_label_emd(train),
    }
    summary_path = write_json(summary, cfg.reports_dir / "dataset_summary.json")
    go_no_go_path = _write_go_no_go(cfg.reports_dir / "go_no_go.md", summary)
    report_paths.update(
        {
            "photo_contact_sheet": contact_sheet,
            "dataset_summary": summary_path,
            "go_no_go": go_no_go_path,
        }
    )
    return report_paths


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_go_no_go(path: Path, summary: dict[str, object]) -> Path:
    ensure_dir(path.parent)
    text = f"""# Go / No-Go Check

## Data Coverage

- Train samples: {_fmt(summary["train_samples"])}
- Test samples: {_fmt(summary["test_samples"])}
- Train photos: {_fmt(summary["train_photos"])}
- Test photos: {_fmt(summary["test_photos"])}
- Unmatched photos: {_fmt(summary["unmatched_photos"])}
- Cameras inferred from filenames: {_fmt(summary["camera_count"])}

## Image-Blind Baselines

- Equal-bin train score: {_fmt(summary["equal_bin_train_score"])}
- Leave-one-out mean score: {_fmt(summary["mean_leave_one_out_score"])}
- Leave-one-out median score: {_fmt(summary["median_leave_one_out_score"])}
- Mean pairwise label EMD: {_fmt(summary["mean_pairwise_label_emd"])}

## Decision Notes

Proceed to image embeddings if:

- photo coverage is complete or nearly complete,
- unmatched photos are explainable from filenames,
- the label curves show meaningful variety,
- the contact sheet suggests visible differences between fine, sandy, and gravelly samples,
- a simple image model can later beat the leave-one-out mean or median baseline under grouped validation.

Be cautious if the mean/median baselines are already close to the pairwise label diversity, or if the photos do not visibly separate the grain-size regimes.
"""
    path.write_text(text, encoding="utf-8")
    return path
