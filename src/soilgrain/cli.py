from __future__ import annotations

import argparse

import pandas as pd

from soilgrain.config import load_config
from soilgrain.submission import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Soil grain Kaggle scaffold")
    parser.add_argument(
        "command",
        choices=["download", "data", "labels", "baselines", "eda", "validate"],
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
    elif args.command == "validate":
        if not args.submission:
            raise SystemExit("--submission is required for validate")
        cfg = load_config(args.config)
        sample_path = cfg.curated_file("sample_submission") if cfg.curated_file("sample_submission").exists() else cfg.raw_file("sample_submission")
        validate_submission(pd.read_csv(args.submission), pd.read_csv(sample_path))
        print(f"Valid submission: {args.submission}")


if __name__ == "__main__":
    main()
