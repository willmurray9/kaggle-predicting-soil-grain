from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from soilgrain.constants import SUPPORT_DIAMETERS


@dataclass(frozen=True)
class ProjectConfig:
    competition: str
    raw_dir: Path
    curated_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    submissions_dir: Path
    files: dict[str, str]
    photo_dirs: dict[str, str]

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/data.yaml") -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            payload: dict[str, Any] = yaml.safe_load(f) or {}
        paths = payload.get("paths", {})
        return cls(
            competition=str(payload.get("competition", "soil-grain-size-from-photos")),
            raw_dir=Path(paths.get("raw_dir", "data/raw/latest")),
            curated_dir=Path(paths.get("curated_dir", "data/curated/latest")),
            artifacts_dir=Path(paths.get("artifacts_dir", "artifacts")),
            reports_dir=Path(paths.get("reports_dir", "artifacts/reports")),
            submissions_dir=Path(paths.get("submissions_dir", "artifacts/submissions")),
            files={k: str(v) for k, v in payload.get("files", {}).items()},
            photo_dirs={k: str(v) for k, v in payload.get("photo_dirs", {}).items()},
        )

    @property
    def support_diameters(self) -> tuple[float, ...]:
        return SUPPORT_DIAMETERS

    def raw_file(self, key: str) -> Path:
        return self.raw_dir / self.files[key]

    def curated_file(self, key: str) -> Path:
        return self.curated_dir / self.files[key]


def load_config(path: str | Path = "configs/data.yaml") -> ProjectConfig:
    return ProjectConfig.from_yaml(path)
