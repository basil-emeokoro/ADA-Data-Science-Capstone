from __future__ import annotations

import pandas as pd


def scenario_targets(paper_count: int) -> list[str]:
    return [f"p{idx}_score" for idx in range(1, paper_count + 1)]


def build_features_for_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    paper_count = int(df["paper_count"].iloc[0])
    candidate_score_cols = [f"p{idx}_score" for idx in range(1, paper_count + 1) if f"p{idx}_score" != target]
    base_cols = candidate_score_cols + [
        "partial_total",
        "mean_score",
        "score_spread",
        "score_std",
        "mean_normalized_score",
    ]
    normalized_cols = [f"p{idx}_normalized" for idx in range(1, paper_count + 1) if f"p{idx}_score" != target]
    feature_cols = [column for column in base_cols + normalized_cols if column in df.columns]
    model_frame = df[feature_cols + [target]].dropna()
    return model_frame[feature_cols], model_frame[target]
