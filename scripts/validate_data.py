from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_validation.clean import clean_data
from src.data_validation.report import write_report
from src.data_validation.schema import build_schema
from src.data_validation.validators import run_validations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fraud dataset.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--out_dir", default="runs", help="Output directory for reports")
    parser.add_argument(
        "--mode",
        default="warn",
        choices=["strict", "warn"],
        help="Validation mode",
    )
    parser.add_argument(
        "--write_cleaned",
        action="store_true",
        help="Write cleaned train/test and validation features",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Loading train data from %s", args.train)
    train_df = pd.read_csv(args.train, low_memory=False)
    logging.info("Loading test data from %s", args.test)
    test_df = pd.read_csv(args.test, low_memory=False)

    schema = build_schema(train_df, test_df, mode=args.mode)
    results = run_validations(train_df, test_df, schema)
    report_paths = write_report(results, args.out_dir)
    logging.info("Wrote report to %s and %s", report_paths["md"], report_paths["json"])

    if args.write_cleaned:
        clean_train, clean_test, validation_features = clean_data(
            train_df, test_df, schema
        )
        clean_train_path = f"{args.out_dir}/clean_train.csv"
        clean_test_path = f"{args.out_dir}/clean_test.csv"
        features_path = f"{args.out_dir}/validation_features.csv"
        clean_train.to_csv(clean_train_path, index=False)
        clean_test.to_csv(clean_test_path, index=False)
        validation_features.to_csv(features_path, index=False)
        logging.info("Wrote cleaned outputs to %s", args.out_dir)

    failures = [c for c in results.get("checks", []) if c.get("status") == "fail"]
    if args.mode == "strict" and failures:
        logging.error("Strict mode failures detected: %d", len(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
