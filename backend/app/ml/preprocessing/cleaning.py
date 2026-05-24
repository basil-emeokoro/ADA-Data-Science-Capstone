from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend.app.schemas.pipeline import MaxScoreMetadata, PaperCountMetadata


MISSING_MARKERS = {"", "ab", "abs", "a", "nan", "-99", "missing", "none", "null"}
PAPER_SCORE_COLUMNS = ["p1_score", "p2_score", "p3_score", "p4_score"]
PAPER_MAX_COLUMNS = ["p1_max", "p2_max", "p3_max", "p4_max"]


@dataclass
class CleaningResult:
    data: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def clean_dataset(
    df: pd.DataFrame,
    detected_max_scores: dict[str, float | None] | None = None,
    paper_counts: list[PaperCountMetadata] | None = None,
    max_scores: list[MaxScoreMetadata] | None = None,
    require_complete_scores: bool = False,
) -> CleaningResult:
    """Convert uploaded data into the frozen canonical schema."""

    warnings: list[str] = []
    errors: list[str] = []
    data = df.copy()
    data = data.drop_duplicates()

    if "anonymized_candidate_id" not in data.columns:
        data["anonymized_candidate_id"] = [f"CAND_{idx:06d}" for idx in range(1, len(data) + 1)]
    if "subject_code" not in data.columns:
        data["subject_code"] = None
    if "subject_name" not in data.columns:
        data["subject_name"] = "Unknown Subject"

    for column in PAPER_SCORE_COLUMNS + PAPER_MAX_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = _to_numeric_score(data[column])

    data["paper_count"] = data.apply(lambda row: _resolve_paper_count(row, paper_counts or []), axis=1)
    if data["paper_count"].isna().any():
        errors.append("Paper count could not be inferred for one or more subjects. Provide metadata recovery input.")
        return CleaningResult(data=data, warnings=warnings, errors=errors)
    data["paper_count"] = data["paper_count"].astype(int)

    _apply_max_scores(data, detected_max_scores or {}, max_scores or [])
    missing_max = _missing_required_maxima(data)
    if missing_max:
        errors.append(f"Maximum scores are missing for required papers: {', '.join(sorted(missing_max))}.")
        return CleaningResult(data=data, warnings=warnings, errors=errors)

    invalid_mask = pd.Series(False, index=data.index)
    for paper_idx in range(1, 5):
        score_col = f"p{paper_idx}_score"
        max_col = f"p{paper_idx}_max"
        active = data["paper_count"] >= paper_idx
        invalid_mask |= active & data[score_col].notna() & data[max_col].notna() & (data[score_col] > data[max_col])
    if invalid_mask.any():
        errors.append("Score validation failed: one or more paper scores exceed their maximum score.")
        return CleaningResult(data=data.loc[~invalid_mask].copy(), warnings=warnings, errors=errors)

    if require_complete_scores:
        missing_scores = pd.Series(False, index=data.index)
        for paper_idx in range(1, 5):
            score_col = f"p{paper_idx}_score"
            active = data["paper_count"] >= paper_idx
            missing_scores |= active & data[score_col].isna()
        if missing_scores.any():
            errors.append("Mode A benchmarking requires complete scores before experimental hiding.")
            return CleaningResult(data=data.loc[~missing_scores].copy(), warnings=warnings, errors=errors)

    data = add_engineered_features(data)
    warnings.extend(_outlier_warnings(data))
    return CleaningResult(data=_canonical_order(data), warnings=warnings, errors=errors)


def _to_numeric_score(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.str.lower().isin(MISSING_MARKERS), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _resolve_paper_count(row: pd.Series, metadata: list[PaperCountMetadata]) -> int | None:
    code = None if pd.isna(row.get("subject_code")) else str(row.get("subject_code")).strip()
    if code and code.lower() not in MISSING_MARKERS:
        last_digit = code[-1]
        if last_digit in {"2", "3", "4"}:
            return int(last_digit)
    subject_name = str(row.get("subject_name", "")).strip().lower()
    for item in metadata:
        if item.subject_code and code and str(item.subject_code).strip() == code:
            return item.paper_count
        if item.subject_name and item.subject_name.strip().lower() == subject_name:
            return item.paper_count
    return None


def _apply_max_scores(
    data: pd.DataFrame,
    detected_max_scores: dict[str, float | None],
    metadata: list[MaxScoreMetadata],
) -> None:
    for column, value in detected_max_scores.items():
        if column in PAPER_MAX_COLUMNS and value is not None:
            data[column] = data[column].fillna(float(value))
    for item in metadata:
        mask = pd.Series(True, index=data.index)
        if item.subject_code:
            mask &= data["subject_code"].astype(str).str.strip() == str(item.subject_code).strip()
        if item.subject_name:
            mask &= data["subject_name"].astype(str).str.strip().str.lower() == item.subject_name.strip().lower()
        for column in PAPER_MAX_COLUMNS:
            value = getattr(item, column)
            if value is not None:
                data.loc[mask, column] = data.loc[mask, column].fillna(float(value))


def _missing_required_maxima(data: pd.DataFrame) -> set[str]:
    missing: set[str] = set()
    for paper_idx in range(1, 5):
        max_col = f"p{paper_idx}_max"
        active = data["paper_count"] >= paper_idx
        if active.any() and data.loc[active, max_col].isna().any():
            missing.add(max_col)
    return missing


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    normalized_cols: list[str] = []
    for paper_idx in range(1, 5):
        score_col = f"p{paper_idx}_score"
        max_col = f"p{paper_idx}_max"
        norm_col = f"p{paper_idx}_normalized"
        output[norm_col] = output[score_col] / output[max_col]
        output.loc[output["paper_count"] < paper_idx, norm_col] = np.nan
        normalized_cols.append(norm_col)

    output["partial_total"] = output[PAPER_SCORE_COLUMNS].sum(axis=1, skipna=True)
    output["mean_score"] = output[PAPER_SCORE_COLUMNS].mean(axis=1, skipna=True)
    output["score_spread"] = output[PAPER_SCORE_COLUMNS].max(axis=1, skipna=True) - output[PAPER_SCORE_COLUMNS].min(axis=1, skipna=True)
    output["score_std"] = output[PAPER_SCORE_COLUMNS].std(axis=1, skipna=True).fillna(0)
    output["mean_normalized_score"] = output[normalized_cols].mean(axis=1, skipna=True)
    return output


def _outlier_warnings(data: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for column in PAPER_SCORE_COLUMNS:
        values = data[column].dropna()
        if len(values) < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        if count:
            warnings.append(f"{count} potential outlier(s) flagged in {column}; values were not removed.")
    return warnings


def _canonical_order(data: pd.DataFrame) -> pd.DataFrame:
    first = [
        "anonymized_candidate_id",
        "subject_code",
        "subject_name",
        "paper_count",
        "p1_score",
        "p2_score",
        "p3_score",
        "p4_score",
        "p1_max",
        "p2_max",
        "p3_max",
        "p4_max",
    ]
    rest = [column for column in data.columns if column not in first]
    return data[first + rest]
