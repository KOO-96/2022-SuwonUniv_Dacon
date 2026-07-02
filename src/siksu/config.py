from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "outputs" / "submissions"

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"
DEFAULT_SUBMISSION_FILE = SUBMISSION_DIR / "submission.csv"

DAY_OF_WEEK_MAP = {
    "월": 1,
    "화": 2,
    "수": 3,
    "목": 4,
    "금": 5,
}

FEATURE_COLUMNS = [
    "요일",
    "월",
    "일",
    "현재원",
    "본사휴가자수",
    "본사출장자수",
    "본사시간외근무명령서승인건수",
    "현본사소속재택근무자수",
]

TARGET_COLUMNS = ["중식계", "석식계"]

DEFAULT_PARAM_GRID = {
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 200, 300],
    "colsample_bytree": [0.7, 0.9, 1.0],
}
