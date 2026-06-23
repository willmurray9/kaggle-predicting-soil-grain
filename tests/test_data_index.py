from pathlib import Path

import pandas as pd

from soilgrain.data import index_photos


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
