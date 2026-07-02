from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from siksu.config import DAY_OF_WEEK_MAP, FEATURE_COLUMNS

SOURCE_COLUMNS = [
    "일자",
    "요일",
    "본사정원수",
    "본사휴가자수",
    "본사출장자수",
    "본사시간외근무명령서승인건수",
    "현본사소속재택근무자수",
]


def validate_columns(frame: pd.DataFrame, required_columns: Iterable[str], frame_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required columns: {missing_text}")


def encode_weekday(weekday: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(weekday):
        return weekday.astype(int)

    normalized = weekday.astype(str).str.strip()
    unknown_values = sorted(set(normalized.dropna()) - set(DAY_OF_WEEK_MAP))
    if unknown_values:
        unknown_text = ", ".join(unknown_values)
        raise ValueError(f"Unknown weekday values: {unknown_text}")

    return normalized.map(DAY_OF_WEEK_MAP).astype(int)


def add_basic_features(frame: pd.DataFrame, frame_name: str = "data") -> pd.DataFrame:
    validate_columns(frame, SOURCE_COLUMNS, frame_name)

    featured = frame.copy()
    dates = pd.to_datetime(featured["일자"], errors="coerce")
    if dates.isna().any():
        invalid_count = int(dates.isna().sum())
        raise ValueError(f"{frame_name} has {invalid_count} invalid values in the 일자 column")

    featured["요일"] = encode_weekday(featured["요일"])
    featured["월"] = dates.dt.month.astype(int)
    featured["일"] = dates.dt.day.astype(int)
    featured["현재원"] = (
        featured["본사정원수"]
        - featured["본사휴가자수"]
        - featured["본사출장자수"]
        - featured["현본사소속재택근무자수"]
    )

    return featured


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    frame_name: str = "data",
) -> pd.DataFrame:
    columns = feature_columns or FEATURE_COLUMNS
    validate_columns(frame, columns, frame_name)

    features = frame.loc[:, columns].copy()
    for column in columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    if features.isna().any().any():
        bad_columns = features.columns[features.isna().any()].tolist()
        bad_text = ", ".join(bad_columns)
        raise ValueError(f"{frame_name} contains non-numeric feature values in: {bad_text}")

    return features


def prepare_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return add_basic_features(train, "train"), add_basic_features(test, "test")
