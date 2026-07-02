from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from siksu.config import (
    DEFAULT_SUBMISSION_FILE,
    FEATURE_COLUMNS,
    SAMPLE_SUBMISSION_FILE,
    TEST_FILE,
    TRAIN_FILE,
)
from siksu.features import build_feature_matrix, prepare_train_test, validate_columns
from siksu.modeling import fit_regressor, model_summary


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    feature_columns: list[str]
    lunch_rows: int
    dinner_rows: int
    lunch_model_summary: dict[str, Any]
    dinner_model_summary: dict[str, Any]


def load_competition_data(
    train_path: Path = TRAIN_FILE,
    test_path: Path = TEST_FILE,
    sample_submission_path: Path = SAMPLE_SUBMISSION_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_submission_path)
    return train, test, sample_submission


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    use_grid_search: bool = True,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], int, int]:
    validate_columns(train, ["중식계", "석식계"], "train")
    train_features, test_features = prepare_train_test(train, test)

    test_matrix = build_feature_matrix(test_features, FEATURE_COLUMNS, "test")

    lunch_matrix = build_feature_matrix(train_features, FEATURE_COLUMNS, "train")
    lunch_model = fit_regressor(
        lunch_matrix,
        train_features["중식계"],
        use_grid_search=use_grid_search,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    lunch_predictions = lunch_model.predict(test_matrix)

    dinner_train = train_features.loc[train_features["석식계"] > 0].copy()
    if dinner_train.empty:
        raise ValueError("No dinner training rows are left after filtering 석식계 > 0")

    dinner_matrix = build_feature_matrix(dinner_train, FEATURE_COLUMNS, "dinner_train")
    dinner_model = fit_regressor(
        dinner_matrix,
        dinner_train["석식계"],
        use_grid_search=use_grid_search,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    dinner_predictions = dinner_model.predict(test_matrix)

    predictions = pd.DataFrame(
        {
            "중식계": np.clip(lunch_predictions, 0, None),
            "석식계": np.clip(dinner_predictions, 0, None),
        }
    )

    return (
        predictions,
        model_summary(lunch_model),
        model_summary(dinner_model),
        len(lunch_matrix),
        len(dinner_matrix),
    )


def create_submission(sample_submission: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    validate_columns(sample_submission, ["중식계", "석식계"], "sample_submission")
    if len(sample_submission) != len(predictions):
        raise ValueError(
            "sample_submission and predictions have different row counts: "
            f"{len(sample_submission)} != {len(predictions)}"
        )

    submission = sample_submission.copy()
    submission["중식계"] = predictions["중식계"].to_numpy()
    submission["석식계"] = predictions["석식계"].to_numpy()
    return submission


def run_pipeline(
    train_path: Path = TRAIN_FILE,
    test_path: Path = TEST_FILE,
    sample_submission_path: Path = SAMPLE_SUBMISSION_FILE,
    output_path: Path = DEFAULT_SUBMISSION_FILE,
    *,
    use_grid_search: bool = True,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> PipelineResult:
    train, test, sample_submission = load_competition_data(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_submission_path,
    )
    predictions, lunch_summary, dinner_summary, lunch_rows, dinner_rows = train_and_predict(
        train,
        test,
        use_grid_search=use_grid_search,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    submission = create_submission(sample_submission, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    return PipelineResult(
        output_path=output_path,
        feature_columns=FEATURE_COLUMNS,
        lunch_rows=lunch_rows,
        dinner_rows=dinner_rows,
        lunch_model_summary=lunch_summary,
        dinner_model_summary=dinner_summary,
    )
