import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage

from soilgrain.config import load_config
from soilgrain.io import write_csv, write_json
from soilgrain.particle_regions import physical_source_crop, particle_regions, region_measurements


PANEL_SOILS = ("F827", "H183", "H516", "H668")
PANEL_CAMERAS = ("Motorola Edge", "Samsung A52")


def _plot_pair(soil, views, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
    for row, (camera, rgb, ppm, labels, seeds, regions) in enumerate(views):
        extent = (0, rgb.shape[1] / ppm, rgb.shape[0] / ppm, 0)
        for ax in axes[row]:
            ax.imshow(rgb, extent=extent)
            ax.set(xlabel="mm", ylabel="mm")
        axes[row, 0].set_title(f"{camera}: source crop ({rgb.shape[1]} pixels)")
        coarse = regions.loc[regions["coarse_interior"], "region_id"].to_numpy()
        truncated = regions.loc[regions["touches_border"], "region_id"].to_numpy()
        boundary = ndimage.maximum_filter(labels, size=3) != ndimage.minimum_filter(labels, size=3)
        boundary[[0, -1], :] |= labels[[0, -1], :] > 0
        boundary[:, [0, -1]] |= labels[:, [0, -1]] > 0
        overlay = np.full((*labels.shape, 4), .7)
        overlay[np.isin(labels, coarse), :3] = (0, .95, 1)
        overlay[np.isin(labels, truncated), :3] = (1, .1, .75)
        overlay[:, :, 3] = boundary * .85
        ax = axes[row, 1]
        seed_overlay = np.empty((*labels.shape, 4))
        seed_overlay[:, :, :3] = (1, .67, .2)
        seed_overlay[:, :, 3] = (seeds > 0) * .4
        ax.imshow(seed_overlay, extent=extent)
        ax.imshow(overlay, extent=extent)
        for region_id in regions.loc[regions["coarse_interior"]].nlargest(6, "area_pixels")["region_id"]:
            coordinates = np.argwhere(labels == region_id)
            center = coordinates.mean(axis=0)
            y, x = coordinates[np.square(coordinates - center).sum(axis=1).argmin()]
            ax.text((x + .5) / ppm, (y + .5) / ppm, str(region_id), fontsize=8, color="white",
                    bbox={"facecolor": "black", "alpha": .6, "pad": 1, "edgecolor": "none"})
        ax.set_title(f"{len(regions)} regions; {len(coarse)} interior candidates ≥2 mm")
    fig.suptitle(f"{soil}: source images and fixed watershed regions — not validated grains")
    fig.supxlabel("Cyan: interior ≥2 mm | Magenta: truncated | Gray: smaller | Orange: seed pixels\nIDs mark the six largest interior candidates; camera views are not registered.", fontsize=9)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_particle_audit(config_path="configs/data.yaml"):
    cfg = load_config(config_path)
    index_path, ppm_path = cfg.reports_dir / "photo_index.csv", cfg.curated_file("ppm")
    manifest_path = cfg.artifacts_dir / "experiments/patch_mixture/summary.json"
    manifest = json.loads(manifest_path.read_text())
    sources = [Path(config_path), index_path, ppm_path, manifest_path]
    recorded = {Path(item["path"]).resolve() for item in manifest["sources"]}
    if not {p.resolve() for p in sources[:-1]} <= recorded:
        raise ValueError("Required source fingerprint is missing")
    for item in [*manifest["sources"], *manifest["photo_sources"]]:
        if sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Recorded source changed: {item['path']}")
    keys = ["split", "sample_id", "camera", "path"]
    index = pd.read_csv(index_path)[keys]
    if index.empty or index[keys].isna().any().any() or index["path"].duplicated().any():
        raise ValueError("Photo index requires complete unique paths")
    ppm = pd.read_csv(ppm_path)
    if ppm["phone"].isna().any() or ppm["phone"].duplicated().any():
        raise ValueError("Calibration requires unique complete phones")
    cameras = ppm.set_index("phone")
    selected = []
    for soil in PANEL_SOILS:
        for camera in PANEL_CAMERAS:
            matches = index.loc[(index["split"] == "train") & (index["sample_id"] == soil) & (index["camera"] == camera)]
            if matches.empty:
                raise ValueError(f"Fixed panel is missing {soil}, {camera}")
            selected.append(matches.sort_values("path").iloc[0])
    photo_records = {item["path"]: item for item in manifest["photo_sources"]}
    if not {row["path"] for row in selected} <= set(photo_records):
        raise ValueError("Panel photo fingerprints are missing")

    output = cfg.artifacts_dir / "experiments/particle_audit"
    output.mkdir(parents=True, exist_ok=True)
    all_regions, photos, outputs, overlays = [], [], [], []
    for soil in PANEL_SOILS:
        views = []
        for row in (row for row in selected if row["sample_id"] == soil):
            camera, source_path = row["camera"], row["path"]
            rgb, effective_ppm = physical_source_crop(source_path, cameras.loc[camera])
            labels, seeds = particle_regions(rgb, effective_ppm)
            regions = region_measurements(labels, effective_ppm)
            stem = f"{soil}_{camera.replace(' ', '_')}"
            crop_path, arrays_path = output / f"{stem}_crop.png", output / f"{stem}_regions.npz"
            Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(crop_path)
            np.savez_compressed(arrays_path, labels=labels, seeds=seeds)
            outputs.extend([crop_path, arrays_path])
            with Image.open(source_path) as source:
                width, height = ImageOps.exif_transpose(source).size
            photos.append({"sample_id": soil, "camera": camera, "source_path": source_path,
                           "source_width": width, "source_height": height, "effective_ppm": effective_ppm,
                           "crop_pixels": rgb.shape[0], "crop_left": (width - rgb.shape[1]) // 2,
                           "crop_top": (height - rgb.shape[0]) // 2, "crop_path": str(crop_path), "arrays_path": str(arrays_path),
                           "regions": len(regions), "coarse_interior_regions": int(regions["coarse_interior"].sum()),
                           "truncated_regions": int(regions["touches_border"].sum()),
                           "coarse_interior_area_fraction": float(regions.loc[regions["coarse_interior"], "area_pixels"].sum() / labels.size)})
            views.append((camera, rgb, effective_ppm, labels, seeds, regions))
            regions = regions.assign(sample_id=soil, camera=camera, source_path=source_path)
            all_regions.append(regions)
        overlay_path = output / f"{soil}_overlay.png"
        _plot_pair(soil, views, overlay_path)
        overlays.append(str(overlay_path)); outputs.append(overlay_path)
        print(f"{soil}: audit overlays saved", flush=True)
    paths = {"photos": write_csv(pd.DataFrame(photos), output / "photos.csv"),
             "regions": write_csv(pd.concat(all_regions, ignore_index=True), output / "regions.csv")}
    outputs.extend(paths.values())
    paths["summary"] = write_json({
        "scope": "Image-only feasibility audit; no soil targets, test images, fitting or submission",
        "review_status": "Pending visual review; regions are not model features",
        "panel_soils": list(PANEL_SOILS), "panel_cameras": list(PANEL_CAMERAS),
        "selection": "Positions 0,7,14,20 among sorted 21 paired training soils; first path per camera",
        "crop_mm": 100, "resize": False, "gradient_sigma": "max(0.5 pixel, 0.25 mm * effective_ppm)",
        "seed_gradient_percentile": 25, "seed_quantile_method": "linear", "opening_disk_radius_mm": .5,
        "connectivity": 8, "watershed_elevation": "round(gradient/max * 65535), uint16; positive seeds only",
        "minimum_interior_diameter_mm": 2, "diameter_definition": "2*sqrt(region area in mm2 / pi); not sieve diameter",
        "photos": photos, "overlays": overlays,
        "sources": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources],
        "photo_sources": [photo_records[row["path"]] for row in selected],
        "outputs": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in outputs],
        "versions": {name: version(name) for name in ("numpy", "pandas", "pillow", "scipy", "matplotlib")},
    }, output / "summary.json")
    return paths
