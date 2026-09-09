from pathlib import Path

import pandas as pd
import pytest

from soilgrain.data import index_photos, load_raw_tables
from soilgrain.config import ProjectConfig


def test_photo_index_maps_images_to_known_sample_ids(tmp_path: Path) -> None:
    train_dir = tmp_path / "All_Photos_training"
    test_dir = tmp_path / "All_Photos_test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "F827_PhoneA_surface.jpg").write_bytes(b"fake")
    (test_dir / "TEST_01_PhoneA_surface.jpg").write_bytes(b"fake")
    ppm = pd.DataFrame({"camera": ["Phone A"], "ppm": [12.5]})

    index = index_photos(
        train_photo_dir=train_dir,
        test_photo_dir=test_dir,
        train_ids=["F827"],
        test_ids=["TEST_01"],
        ppm=ppm,
    )

    assert set(index["sample_id"]) == {"F827", "TEST_01"}
    assert set(index["split"]) == {"train", "test"}


def test_photo_index_infers_camera_from_ppm_table(tmp_path: Path) -> None:
    train_dir = tmp_path / "All_Photos_training"
    test_dir = tmp_path / "All_Photos_test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "F827_PhoneA_surface.jpg").write_bytes(b"fake")
    ppm = pd.DataFrame({"camera": ["Phone A", "Other Camera"], "ppm": [12.5, 3.2]})

    index = index_photos(
        train_photo_dir=train_dir,
        test_photo_dir=test_dir,
        train_ids=["F827"],
        test_ids=[],
        ppm=ppm,
    )

    assert index.loc[0, "camera"] == "Phone A"
    assert index.loc[0, "ppm"] == 12.5


def test_photo_index_matches_phone_name_when_camera_model_differs(tmp_path: Path) -> None:
    train_dir = tmp_path / "All_Photos_training"
    test_dir = tmp_path / "All_Photos_test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "Samsung_A52_F827_01.jpg").write_bytes(b"fake")
    ppm = pd.DataFrame(
        {"camera": ["SM-A525F"], "phone": ["Samsung A52"], "ppm": [26.33]}
    )

    index = index_photos(train_dir, test_dir, ["F827"], [], ppm)

    assert index.loc[0, "camera"] == "Samsung A52"
    assert index.loc[0, "ppm"] == 26.33


def test_photo_index_normalizes_german_umlaut_and_separators(tmp_path: Path) -> None:
    train_dir = tmp_path / "All_Photos_training"
    test_dir = tmp_path / "All_Photos_test"
    train_dir.mkdir()
    test_dir.mkdir()
    (test_dir / "iPhone14_HPC_Münster_BS6_9,0-10m (1).JPG").write_bytes(b"fake")

    index = index_photos(
        train_dir,
        test_dir,
        [],
        ["HPC_Muenster_BS6_9_0-10m"],
        pd.DataFrame({"phone": ["iPhone 14"], "ppm": [13.942]}),
    )

    assert index.loc[0, "sample_id"] == "HPC_Muenster_BS6_9_0-10m"


def test_photo_index_rejects_ambiguous_sample_ids(tmp_path: Path) -> None:
    train_dir = tmp_path / "All_Photos_training"
    test_dir = tmp_path / "All_Photos_test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "sample_alpha_beta.jpg").write_bytes(b"fake")

    with pytest.raises(ValueError, match="ambiguous sample ID"):
        index_photos(
            train_dir,
            test_dir,
            ["sample_alpha", "alpha_beta"],
            [],
            pd.DataFrame(),
        )


def test_load_raw_tables_derives_test_ids_from_submission(tmp_path: Path) -> None:
    pd.DataFrame({"sample_id": ["TRAIN_1"], "0.002": [10.0]}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    pd.DataFrame({"sample_id": ["TEST_1"], "0.002": [0.0]}).to_csv(
        tmp_path / "submission.csv", index=False
    )
    pd.DataFrame({"phone": ["Phone A"], "ppm": [12.5]}).to_csv(
        tmp_path / "ppm.csv", index=False
    )
    cfg = ProjectConfig(
        competition="test",
        raw_dir=tmp_path,
        curated_dir=tmp_path / "curated",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        submissions_dir=tmp_path / "submissions",
        files={
            "train": "labels.csv",
            "sample_submission": "submission.csv",
            "ppm": "ppm.csv",
        },
        photo_dirs={"training": "training", "test": "test"},
    )

    _train, test, _sample, _ppm = load_raw_tables(cfg)

    assert test.to_dict(orient="records") == [{"sample_id": "TEST_1"}]


def test_photo_index_reports_unmatched_photos(tmp_path: Path) -> None:
    train_dir = tmp_path / "All_Photos_training"
    test_dir = tmp_path / "All_Photos_test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "mystery_surface.jpg").write_bytes(b"fake")
    ppm = pd.DataFrame({"camera": ["Phone A"], "ppm": [12.5]})

    index = index_photos(
        train_photo_dir=train_dir,
        test_photo_dir=test_dir,
        train_ids=["F827"],
        test_ids=[],
        ppm=ppm,
    )

    assert index.loc[0, "sample_id"] == ""
    assert index.loc[0, "issue"] == "unmatched_sample_id"
