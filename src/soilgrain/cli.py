from __future__ import annotations

import argparse

import pandas as pd

from soilgrain.config import load_config
from soilgrain.submission import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Soil grain Kaggle scaffold")
    parser.add_argument(
        "command",
        choices=["download", "data", "labels", "baselines", "eda", "image-model", "experiments", "multicrop", "audit", "camera-balance", "kernel-ridge", "frozen-model", "nested-ridge", "model-blend", "nested-neighbors", "spatial-coverage", "validate"],
    )
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--submission", help="Submission CSV to validate.")
    args = parser.parse_args()

    if args.command == "download":
        from soilgrain.data import download_competition_data

        download_competition_data(args.config)
    elif args.command == "data":
        from soilgrain.data import prepare_data

        index = prepare_data(args.config)
        print(f"Wrote photo index with {len(index)} rows.")
    elif args.command == "labels":
        from soilgrain.eda import write_label_reports

        paths = write_label_reports(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "baselines":
        from soilgrain.baselines import write_baseline_submissions

        paths = write_baseline_submissions(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "eda":
        from soilgrain.eda import write_eda_reports

        paths = write_eda_reports(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "image-model":
        from soilgrain.image_model import write_image_model

        paths = write_image_model(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "experiments":
        from soilgrain.experiments import write_experiments

        paths = write_experiments(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "multicrop":
        from soilgrain.multicrop import write_multicrop

        paths = write_multicrop(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "audit":
        from soilgrain.audit import write_audit

        paths = write_audit(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "camera-balance":
        from soilgrain.camera_balance import write_camera_balance

        paths = write_camera_balance(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "kernel-ridge":
        from soilgrain.kernel_experiment import write_kernel_experiment

        paths = write_kernel_experiment(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "frozen-model":
        from soilgrain.frozen_experiment import write_frozen_experiment

        paths = write_frozen_experiment(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "nested-ridge":
        from soilgrain.nested_experiment import write_nested_experiment

        paths = write_nested_experiment(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "model-blend":
        from soilgrain.model_blend import write_model_blend

        paths = write_model_blend(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "nested-neighbors":
        from soilgrain.neighbor_experiment import write_neighbor_experiment

        paths = write_neighbor_experiment(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "spatial-coverage":
        from soilgrain.spatial_experiment import write_spatial_experiment

        paths = write_spatial_experiment(args.config)
        for path in paths.values():
            print(path)
    elif args.command == "validate":
        if not args.submission:
            raise SystemExit("--submission is required for validate")
        cfg = load_config(args.config)
        sample_path = cfg.curated_file("sample_submission") if cfg.curated_file("sample_submission").exists() else cfg.raw_file("sample_submission")
        validate_submission(pd.read_csv(args.submission), pd.read_csv(sample_path))
        print(f"Valid submission: {args.submission}")


if __name__ == "__main__":
    main()
