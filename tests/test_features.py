from __future__ import annotations

import pandas as pd
import pytest

from siksu.features import add_basic_features, build_feature_matrix, encode_weekday


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "일자": ["2021-01-04", "2021-01-05"],
            "요일": ["월", "화"],
            "본사정원수": [1000, 1000],
            "본사휴가자수": [10, 20],
            "본사출장자수": [30, 40],
            "본사시간외근무명령서승인건수": [100, 110],
            "현본사소속재택근무자수": [50, 60],
        }
    )


def test_add_basic_features_adds_calendar_and_attendance_columns() -> None:
    featured = add_basic_features(make_frame())

    assert featured["요일"].tolist() == [1, 2]
    assert featured["월"].tolist() == [1, 1]
    assert featured["일"].tolist() == [4, 5]
    assert featured["현재원"].tolist() == [910, 880]


def test_build_feature_matrix_returns_numeric_columns() -> None:
    featured = add_basic_features(make_frame())
    matrix = build_feature_matrix(featured)

    assert matrix.shape == (2, 8)
    assert matrix.select_dtypes(include="number").shape[1] == 8


def test_encode_weekday_rejects_unexpected_labels() -> None:
    with pytest.raises(ValueError, match="Unknown weekday values"):
        encode_weekday(pd.Series(["월", "토"]))

