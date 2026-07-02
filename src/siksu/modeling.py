from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV

from siksu.config import DEFAULT_PARAM_GRID


def create_xgb_regressor(random_state: int = 42, n_jobs: int = -1) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required to train the competition model."
        ) from exc

    return XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def fit_regressor(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    use_grid_search: bool = True,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    param_grid: dict[str, list[float | int]] | None = None,
) -> Any:
    estimator = create_xgb_regressor(random_state=random_state, n_jobs=n_jobs)
    if not use_grid_search or len(features) < 2:
        estimator.fit(features, target)
        return estimator

    folds = max(2, min(cv, len(features)))
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid or DEFAULT_PARAM_GRID,
        cv=folds,
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
    )
    search.fit(features, target)
    return search


def model_summary(model: Any) -> dict[str, Any]:
    if hasattr(model, "best_params_"):
        return {
            "model": model.best_estimator_.__class__.__name__,
            "best_params": model.best_params_,
            "best_score": model.best_score_,
        }

    return {"model": model.__class__.__name__}
