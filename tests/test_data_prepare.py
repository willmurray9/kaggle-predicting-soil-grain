import json
from pathlib import Path

import pandas as pd

from soilgrain.data import prepare_data


def test_prepare_data_reports_photo_coverage_gaps(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    train_dir = raw_dir / "training"
    test_dir = raw_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir()
    pd.DataFrame(
        {"sample_id": ["TRAIN_1", "TRAIN_MISSING"], "0.002": [10.0, 20.0]}
    ).to_csv(raw_dir / "labels.csv", index=False)
    pd.DataFrame(
        {"sample_id": ["TEST_1", "TEST_MISSING"], "0.002": [0.0, 0.0]}
    ).to_csv(raw_dir / "submission.csv", index=False)
    pd.DataFrame({"phone": ["Known Phone"], "ppm": [12.5]}).to_csv(
        raw_dir / "ppm.csv", index=False
    )
    (train_dir / "Known_Phone_TRAIN_1.jpg").write_bytes(b"fake")
    (test_dir / "Mystery_TEST_1.jpg").write_bytes(b"fake")
    (test_dir / "Known_Phone_UNLISTED.jpg").write_bytes(b"fake")
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        f"""
competition: test
paths:
  raw_dir: {raw_dir}
  curated_dir: {tmp_path / 'curated'}
  artifacts_dir: {tmp_path / 'artifacts'}
  reports_dir: {tmp_path / 'reports'}
  submissions_dir: {tmp_path / 'submissions'}
files:
  train: labels.csv
  sample_submission: submission.csv
  ppm: ppm.csv
photo_dirs:
  training: training
  test: test
""",
        encoding="utf-8",
    )

    index = prepare_data(config_path)
    summary = json.loads((tmp_path / "reports" / "data_summary.json").read_text())

    assert summary["missing_train_sample_ids"] == ["TRAIN_MISSING"]
    assert summary["missing_test_sample_ids"] == ["TEST_MISSING"]
    assert summary["unmatched_photo_rows"] == 1
    assert summary["unknown_camera_photo_rows"] == 1
    assert index.set_index("filename").loc["Mystery_TEST_1.jpg", "issue"] == "unknown_camera"
