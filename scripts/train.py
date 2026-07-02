#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from siksu.config import DEFAULT_SUBMISSION_FILE, SAMPLE_SUBMISSION_FILE, TEST_FILE, TRAIN_FILE
from siksu.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=SAMPLE_SUBMISSION_FILE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUBMISSION_FILE,
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        train_path=args.train,
        test_path=args.test,
        sample_submission_path=args.sample_submission,
        output_path=args.output,
        use_grid_search=not args.no_grid_search,
        cv=args.cv,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    print(f"Submission saved to: {result.output_path}")
    print(f"Lunch rows: {result.lunch_rows}")
    print(f"Dinner rows: {result.dinner_rows}")
    print(f"Lunch model: {result.lunch_model_summary}")
    print(f"Dinner model: {result.dinner_model_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
