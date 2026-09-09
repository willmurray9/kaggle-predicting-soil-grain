import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgrain.constants import CANONICAL_GRAIN_LABELS
from soilgrain.multicrop import blend_predictions, write_multicrop


COMPONENTS = ("gray_50", "gray_100", "gray_150")


def _frame(ids: list[str], starts: list[float], *, cameras: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"sample_id": ids})
    if cameras is not None:
        frame["camera"] = cameras
    values = np.array([np.linspace(start, 100.0, 11) for start in starts])
    frame[list(CANONICAL_GRAIN_LABELS)] = values
    return frame


def test_blend_predictions_aligns_shuffled_sample_ids() -> None:
    components = {
        "gray_50": _frame(["A", "B"], [0.0, 30.0]),
        "gray_100": _frame(["B", "A"], [60.0, 30.0]),
        "gray_150": _frame(["A", "B"], [60.0, 0.0]),
    }

    blended = blend_predictions(components, ["sample_id"])

    assert blended["sample_id"].tolist() == ["A", "B"]
    np.testing.assert_allclose(blended["0.002"], [30.0, 30.0])


@pytest.mark.parametrize(
    "bad_component",
    [
        _frame(["A"], [0.0]),
        _frame(["A", "A"], [0.0, 30.0]),
    ],
    ids=["dropped-key", "duplicate-key"],
)
def test_blend_predictions_rejects_incomplete_or_duplicate_ids(
    bad_component: pd.DataFrame,
) -> None:
    components = {
        "gray_50": _frame(["A", "B"], [0.0, 30.0]),
        "gray_100": bad_component,
        "gray_150": _frame(["A", "B"], [60.0, 0.0]),
    }

    with pytest.raises(ValueError, match="duplicate|coverage"):
        blend_predictions(components, ["sample_id"])


def test_blend_predictions_requires_all_three_components_and_valid_curves() -> None:
    components = {
        "gray_50": _frame(["A"], [0.0]),
        "gray_100": _frame(["A"], [30.0]),
    }
    with pytest.raises(ValueError, match="exactly"):
        blend_predictions(components, ["sample_id"])

    components["gray_150"] = _frame(["A"], [60.0])
    components["gray_150"].loc[0, "0.0063"] = -1.0
    with pytest.raises(ValueError, match="within"):
        blend_predictions(components, ["sample_id"])

    empty_components = {
        name: _frame(["A"], [30.0]).iloc[0:0] for name in COMPONENTS
    }
    with pytest.raises(ValueError, match="empty"):
        blend_predictions(empty_components, ["sample_id"])


def test_blend_predictions_matches_sample_and_camera_together() -> None:
    components = {
        "gray_50": _frame(["A", "A"], [0.0, 30.0], cameras=["left", "right"]),
        "gray_100": _frame(["A", "A"], [60.0, 0.0], cameras=["right", "left"]),
        "gray_150": _frame(["A", "A"], [30.0, 60.0], cameras=["left", "right"]),
    }

    blended = blend_predictions(components, ["sample_id", "camera"])

    np.testing.assert_allclose(blended["0.002"], [10.0, 50.0])


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "data.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  curated_dir: {tmp_path / 'curated'}",
                f"  artifacts_dir: {tmp_path / 'artifacts'}",
                f"  submissions_dir: {tmp_path / 'submissions'}",
                "files:",
                "  train: train.csv",
                "  sample_submission: sample.csv",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_write_multicrop_scores_averaged_curves_and_writes_provenance(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    source_dir = tmp_path / "artifacts" / "experiments" / "first_batch"
    submission_dir = tmp_path / "submissions" / "experiments"
    curated_dir = tmp_path / "curated"
    source_dir.mkdir(parents=True)
    submission_dir.mkdir(parents=True)
    curated_dir.mkdir()

    truth = _frame(["B", "A"], [20.0, 50.0])
    truth.loc[truth["sample_id"] == "B", list(CANONICAL_GRAIN_LABELS[:-1])] = 20.0
    truth.rename(columns={"2.0": "2", "20.0": "20", "63.0": "63", "200.0": "200"}).to_csv(
        curated_dir / "train.csv", index=False
    )
    sample = truth.drop(columns="sample_id").iloc[[0, 0]].copy()
    sample.insert(0, "sample_id", ["T1", "T2"])
    sample.rename(columns={"2.0": "2", "20.0": "20", "63.0": "63", "200.0": "200"}).to_csv(
        curated_dir / "sample.csv", index=False
    )

    oof_rows = []
    camera_rows = []
    submission_starts = {"gray_50": [0.0, 30.0], "gray_100": [100.0, 60.0], "gray_150": [50.0, 0.0]}
    for experiment, start in zip(COMPONENTS, [0.0, 100.0, 50.0], strict=True):
        oof = _frame(["A", "B"], [start, 0.0])
        oof.loc[oof["sample_id"] == "B", list(CANONICAL_GRAIN_LABELS[:-1])] = 0.0
        oof.insert(0, "experiment", experiment)
        oof.insert(2, "emd", 999.0)
        oof_rows.append(oof)
        camera = _frame(["A", "A"], [start, 100.0 - start], cameras=["left", "right"])
        camera.insert(0, "experiment", experiment)
        camera.insert(3, "emd", 999.0)
        camera_rows.append(camera)
        candidate = _frame(["T1", "T2"], submission_starts[experiment])
        candidate.rename(columns={"2.0": "2", "20.0": "20", "63.0": "63", "200.0": "200"}).to_csv(
            submission_dir / f"{experiment}.csv", index=False
        )
    pd.concat(oof_rows).to_csv(source_dir / "oof_predictions.csv", index=False)
    pd.concat(camera_rows).to_csv(source_dir / "camera_predictions.csv", index=False)

    paths = write_multicrop(config)

    assert paths["submission"] == tmp_path / "submissions" / "multicrop_baseline.csv"
    blended_oof = pd.read_csv(paths["oof"])
    blended_cameras = pd.read_csv(paths["camera_predictions"])
    submission = pd.read_csv(paths["submission"])
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert blended_oof.loc[0, "emd"] == pytest.approx(0.0)
    assert blended_oof.loc[1, "emd"] == pytest.approx(100.0)
    assert summary["loo_emd"] == pytest.approx(50.0)
    assert summary["paired_camera_disagreement_emd"] == pytest.approx(0.0)
    assert blended_cameras["emd"].tolist() == pytest.approx([0.0, 0.0])
    np.testing.assert_allclose(submission["0.002"], [50.0, 30.0])
    assert list(submission.columns) == list(pd.read_csv(curated_dir / "sample.csv").columns)
    assert len(summary["component_sources"]) == 5
    assert all(len(source["sha256"]) == 64 for source in summary["component_sources"])
