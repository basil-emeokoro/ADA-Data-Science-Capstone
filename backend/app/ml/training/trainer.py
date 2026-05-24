from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from backend.app.config.settings import get_settings
from backend.app.ml.evaluation.metrics import regression_metrics
from backend.app.ml.models.registry import get_model_registry


@dataclass
class TrainedScenario:
    subject_key: str
    paper_count: int
    target: str
    best_model_name: str
    best_model: Any
    feature_columns: list[str]
    metrics: list[dict[str, Any]]
    ranking: list[dict[str, Any]]
    y_test: list[float]
    y_pred: list[float]


def train_scenario(subject_key: str, paper_count: int, target: str, X: pd.DataFrame, y: pd.Series) -> TrainedScenario:
    settings = get_settings()
    if len(X) < 6:
        raise ValueError(f"Not enough complete rows to train {target}; at least 6 rows are required.")

    test_size = settings.test_size if len(X) >= 10 else max(1 / len(X), 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=settings.random_state)
    model_results: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    cv_splits = min(settings.cv_folds, len(X_train))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=settings.random_state) if cv_splits >= 2 else None

    for model_name, model in get_model_registry(settings.random_state).items():
        estimator = clone(model)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        cv_rmse = None
        if cv is not None:
            scores = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
            cv_rmse = float(np.abs(scores).mean())
        row = {
            "subject": subject_key,
            "paper_count": paper_count,
            "scenario": f"Hide {target.replace('_score', '').upper()}",
            "target": target,
            "model": model_name,
            **metrics,
            "CV_RMSE": cv_rmse,
        }
        model_results.append(row)
        fitted_models[model_name] = estimator

    ranked = sorted(model_results, key=lambda item: (item["RMSE"], item["MAE"]))
    for rank, row in enumerate(ranked, start=1):
        row["Rank"] = rank
    best_name = ranked[0]["model"]
    best_model = fitted_models[best_name]
    best_pred = best_model.predict(X_test)
    return TrainedScenario(
        subject_key=subject_key,
        paper_count=paper_count,
        target=target,
        best_model_name=best_name,
        best_model=best_model,
        feature_columns=list(X.columns),
        metrics=model_results,
        ranking=ranked,
        y_test=[float(v) for v in y_test.tolist()],
        y_pred=[float(v) for v in best_pred.tolist()],
    )
