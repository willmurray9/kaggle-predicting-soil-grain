from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from soilgrain.config import ProjectConfig, load_config
from soilgrain.constants import IMAGE_EXTENSIONS
from soilgrain.io import ensure_dir, read_csv, write_csv, write_json


def _normalize_token(value: object) -> str:
    folded = str(value).casefold().translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}))
    return "".join(ch for ch in folded if ch.isalnum())


def _match_value_in_filename(filename: str, values: Iterable[object]) -> str:
    lowered = filename.lower()
    compact = _normalize_token(filename)
    for value in sorted((str(v) for v in values), key=len, reverse=True):
        if not value:
            continue
        if value.lower() in lowered or _normalize_token(value) in compact:
            return value
    return ""


def _match_sample_id(filename: str, values: Iterable[object]) -> str:
    compact = _normalize_token(filename)
    matches = [str(value) for value in values if str(value) and _normalize_token(value) in compact]
    if len(matches) > 1:
        raise ValueError(f"{filename!r} has ambiguous sample ID matches: {matches}")
    return matches[0] if matches else ""


def _camera_column(ppm: pd.DataFrame) -> str | None:
    for col in ppm.columns:
        if str(col).lower() == "phone":
            return str(col)
    for col in ppm.columns:
        lower = str(col).lower()
        if "camera" in lower or "device" in lower:
            return str(col)
    object_cols = [str(c) for c in ppm.columns if ppm[c].dtype == "object"]
    return object_cols[0] if object_cols else None


def _ppm_column(ppm: pd.DataFrame) -> str | None:
    for col in ppm.columns:
        if str(col).lower() == "ppm":
            return str(col)
    for col in ppm.columns:
        lower = str(col).lower()
        if "ppm" in lower or ("pixel" in lower and "mm" in lower):
            return str(col)
    numeric_cols = [str(c) for c in ppm.columns if pd.api.types.is_numeric_dtype(ppm[c])]
    return numeric_cols[0] if numeric_cols else None


def _infer_camera(filename: str, ppm: pd.DataFrame) -> tuple[str, float | None]:
    if ppm.empty:
        return "", None
    cam_col = _camera_column(ppm)
    ppm_col = _ppm_column(ppm)
    if cam_col is None:
        return "", None

    camera = _match_value_in_filename(filename, ppm[cam_col].dropna().astype(str))
    if not camera:
        return "", None
    row = ppm[ppm[cam_col].astype(str) == camera]
    ppm_value = None
    if ppm_col is not None and not row.empty:
        parsed = pd.to_numeric(row.iloc[0][ppm_col], errors="coerce")
        ppm_value = None if pd.isna(parsed) else float(parsed)
    return camera, ppm_value


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def index_photos(
    train_photo_dir: str | Path,
    test_photo_dir: str | Path,
    train_ids: Iterable[object],
    test_ids: Iterable[object],
    ppm: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, root, ids in [
        ("train", Path(train_photo_dir), list(train_ids)),
        ("test", Path(test_photo_dir), list(test_ids)),
    ]:
        for path in _image_files(root):
            sample_id = _match_sample_id(path.name, ids)
            camera, ppm_value = _infer_camera(path.name, ppm)
            rows.append(
                {
                    "split": split,
                    "path": str(path),
                    "filename": path.name,
                    "sample_id": sample_id,
                    "camera": camera,
                    "ppm": ppm_value,
                    "is_matched": bool(sample_id),
                    "issue": (
                        "unmatched_sample_id"
                        if not sample_id
                        else "unknown_camera"
                        if not camera
                        else ""
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["split", "path", "filename", "sample_id", "camera", "ppm", "is_matched", "issue"],
    )


def _find_photo_dir(raw_dir: Path, expected_name: str) -> Path:
    direct = raw_dir / expected_name
    if direct.exists():
        return direct
    if raw_dir.exists():
        for path in raw_dir.rglob("*"):
            if path.is_dir() and path.name.lower() == expected_name.lower():
                return path
    raise FileNotFoundError(f"Could not find photo directory named {expected_name!r} under {raw_dir}")


def _require_raw_files(cfg: ProjectConfig) -> None:
    keys = ["train", "sample_submission", "ppm"]
    if "test" in cfg.files:
        keys.append("test")
    missing = [str(cfg.raw_file(key)) for key in keys if not cfg.raw_file(key).exists()]
    if missing:
        raise FileNotFoundError("Missing required Kaggle files:\n" + "\n".join(f"- {m}" for m in missing))


def load_raw_tables(cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_raw_files(cfg)
    sample = read_csv(cfg.raw_file("sample_submission"))
    test = read_csv(cfg.raw_file("test")) if "test" in cfg.files else sample[["sample_id"]].copy()
    return read_csv(cfg.raw_file("train")), test, sample, read_csv(cfg.raw_file("ppm"))


def load_working_tables(cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = cfg.curated_dir if cfg.curated_file("train").exists() else cfg.raw_dir
    sample = read_csv(source / cfg.files["sample_submission"])
    test = read_csv(source / cfg.files["test"]) if "test" in cfg.files else sample[["sample_id"]].copy()
    return read_csv(source / cfg.files["train"]), test, sample, read_csv(source / cfg.files["ppm"])


def prepare_data(config_path: str | Path = "configs/data.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    train, test, sample, ppm = load_raw_tables(cfg)
    for name, df in [("train.csv", train), ("test.csv", test), ("sample_submission.csv", sample)]:
        if "sample_id" not in df.columns:
            raise ValueError(f"{name} must contain sample_id.")

    ensure_dir(cfg.curated_dir)
    tables = [("train", train), ("sample_submission", sample), ("ppm", ppm)]
    if "test" in cfg.files:
        tables.append(("test", test))
    for key, df in tables:
        write_csv(df, cfg.curated_file(key))

    train_dir = _find_photo_dir(cfg.raw_dir, cfg.photo_dirs["training"])
    test_dir = _find_photo_dir(cfg.raw_dir, cfg.photo_dirs["test"])
    photo_index = index_photos(train_dir, test_dir, train["sample_id"], test["sample_id"], ppm)
    write_csv(photo_index, cfg.reports_dir / "photo_index.csv")
    matched_train_ids = set(photo_index.loc[(photo_index["split"] == "train") & photo_index["is_matched"], "sample_id"])
    matched_test_ids = set(photo_index.loc[(photo_index["split"] == "test") & photo_index["is_matched"], "sample_id"])
    write_json(
        {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "photo_rows": int(len(photo_index)),
            "matched_photo_rows": int(photo_index["is_matched"].sum()) if not photo_index.empty else 0,
            "unmatched_photo_rows": int((~photo_index["is_matched"]).sum()) if not photo_index.empty else 0,
            "unknown_camera_photo_rows": int(photo_index["camera"].eq("").sum()) if not photo_index.empty else 0,
            "missing_train_sample_ids": sorted(set(train["sample_id"].astype(str)) - matched_train_ids),
            "missing_test_sample_ids": sorted(set(test["sample_id"].astype(str)) - matched_test_ids),
        },
        cfg.reports_dir / "data_summary.json",
    )
    return photo_index


def _kaggle_executable() -> str:
    candidates = [
        Path(sys.executable).with_name("kaggle"),
        Path("/opt/homebrew/bin/kaggle"),
        Path("/usr/local/bin/kaggle"),
        Path.home() / ".local/bin/kaggle",
        Path("/opt/anaconda3/bin/kaggle"),
    ]
    which = shutil.which("kaggle")
    if which:
        candidates.insert(0, Path(which))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("Could not find the Kaggle CLI. Install it or place files manually in data/raw/latest/.")


def download_competition_data(config_path: str | Path = "configs/data.yaml") -> None:
    cfg = load_config(config_path)
    raw_dir = ensure_dir(cfg.raw_dir)
    cmd = [_kaggle_executable(), "competitions", "download", "-c", cfg.competition, "-p", str(raw_dir), "--force"]
    subprocess.run(cmd, check=True)
    for archive in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
    print(f"Downloaded and extracted Kaggle data to {raw_dir}")
