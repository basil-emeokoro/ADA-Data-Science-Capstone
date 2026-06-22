from __future__ import annotations

import pandas as pd

from backend.app.ml.preprocessing.cleaning import add_engineered_features


def scenario_targets(paper_count: int) -> list[str]:
    return [f"p{idx}_score" for idx in range(1, paper_count + 1)]


def build_features_for_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build scenario features after hiding the target paper.

    Aggregate features must be recomputed from visible papers only. Reusing
    aggregates calculated from the complete record would leak the target.
    """

    paper_count = int(df["paper_count"].iloc[0])
    visible = df.copy()
    visible[target] = float("nan")
    visible = add_engineered_features(visible)
    candidate_score_cols = [f"p{idx}_score" for idx in range(1, paper_count + 1) if f"p{idx}_score" != target]
    base_cols = candidate_score_cols + [
        "partial_total",
        "mean_score",
        "score_spread",
        "score_std",
        "mean_normalized_score",
    ]
    normalized_cols = [f"p{idx}_normalized" for idx in range(1, paper_count + 1) if f"p{idx}_score" != target]
    feature_cols = [column for column in base_cols + normalized_cols if column in visible.columns]
    model_frame = visible[feature_cols].copy()
    model_frame[target] = pd.to_numeric(df[target], errors="coerce")
    model_frame = model_frame.dropna()
    return model_frame[feature_cols], model_frame[target]
