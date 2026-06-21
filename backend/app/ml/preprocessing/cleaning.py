from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend.app.schemas.pipeline import MaxScoreMetadata, PaperCountMetadata


INVALID_MARKERS = {"-99", "b", "null"}
ABSENT_MARKERS = {"a", "ab", "abs"}
PREDICTABLE_MISSING_MARKERS = {"", "missing", "nan", "none"}
METADATA_MISSING_MARKERS = INVALID_MARKERS | ABSENT_MARKERS | PREDICTABLE_MISSING_MARKERS
PAPER_SCORE_COLUMNS = ["p1_score", "p2_score", "p3_score", "p4_score"]
PAPER_MAX_COLUMNS = ["p1_max", "p2_max", "p3_max", "p4_max"]


@dataclass
class CleaningResult:
    data: pd.DataFrame
    invalid_records: pd.DataFrame = field(default_factory=pd.DataFrame)
    absent_records: pd.DataFrame = field(default_factory=pd.DataFrame)
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
    uploaded_score_columns = set(data.columns) & set(PAPER_SCORE_COLUMNS)

    if "anonymized_candidate_id" not in data.columns:
        data["anonymized_candidate_id"] = [f"CAND_{idx:06d}" for idx in range(1, len(data) + 1)]
    if "subject_code" not in data.columns:
        data["subject_code"] = None
    if "subject_id" not in data.columns:
        data["subject_id"] = None
    if "subject_name" not in data.columns:
        data["subject_name"] = "Unknown Subject"

    raw_score_columns = _capture_raw_score_columns(data)
    for column in PAPER_SCORE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
    for column in PAPER_MAX_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = _to_numeric_score(data[column])

    data["paper_count"] = data.apply(lambda row: _resolve_paper_count(row, paper_counts or []), axis=1)
    if data["paper_count"].isna().any():
        errors.append("Paper count could not be inferred for one or more subjects. Provide metadata recovery input.")
        return CleaningResult(data=data, warnings=warnings, errors=errors)
    data["paper_count"] = data["paper_count"].astype(int)

    missing_paper_columns = [
        f"p{paper_idx}_score"
        for paper_idx in range(1, 5)
        if f"p{paper_idx}_score" not in uploaded_score_columns and (data["paper_count"] >= paper_idx).any()
    ]
    if missing_paper_columns:
        errors.append(
            "Missing paper column detected. Keep all applicable paper columns in the file and mark missing candidate "
            "scores as blank/missing. The system needs complete rows to train prediction models. "
            f"Missing columns: {', '.join(missing_paper_columns)}."
        )
        return CleaningResult(data=data, warnings=warnings, errors=errors)

    classification = _classify_score_records(data, raw_score_columns)
    invalid_records = classification["invalid_records"]
    absent_records = classification["absent_records"]
    if not invalid_records.empty:
        warnings.append(f"{len(invalid_records)} invalid record(s) isolated and excluded from training/prediction.")
    if not absent_records.empty:
        warnings.append(f"{len(absent_records)} absent record(s) isolated and excluded from Mode A training.")

    data = classification["cleanable_data"]
    for column in PAPER_SCORE_COLUMNS:
        data[column] = _to_numeric_score(data[column])
    data = _drop_non_applicable_scores(data)

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
        over_max = data.loc[invalid_mask].copy()
        over_max["record_status"] = "invalid"
        over_max["record_reason"] = "score exceeds maximum score"
        invalid_records = _append_records(invalid_records, over_max)
        warnings.append(f"{int(invalid_mask.sum())} over-maximum record(s) isolated as invalid.")
        data = data.loc[~invalid_mask].copy()

    if require_complete_scores:
        missing_scores = pd.Series(False, index=data.index)
        for paper_idx in range(1, 5):
            score_col = f"p{paper_idx}_score"
            active = data["paper_count"] >= paper_idx
            missing_scores |= active & data[score_col].isna()
        if missing_scores.any():
            incomplete = data.loc[missing_scores].copy()
            incomplete["record_status"] = "invalid"
            incomplete["record_reason"] = "incomplete applicable paper scores"
            invalid_records = _append_records(invalid_records, incomplete)
            removed_count = int(missing_scores.sum())
            warnings.append(f"{removed_count} incomplete record(s) isolated before Mode A benchmarking.")
            data = data.loc[~missing_scores].copy()

    data = add_engineered_features(data)
    warnings.extend(_outlier_warnings(data))
    return CleaningResult(
        data=_canonical_order(data),
        invalid_records=_canonical_order_if_possible(invalid_records),
        absent_records=_canonical_order_if_possible(absent_records),
        warnings=warnings,
        errors=errors,
    )


def _capture_raw_score_columns(data: pd.DataFrame) -> dict[str, pd.Series]:
    raw: dict[str, pd.Series] = {}
    for column in PAPER_SCORE_COLUMNS:
        if column in data.columns:
            raw[column] = data[column].astype(str).str.strip()
        else:
            raw[column] = pd.Series("", index=data.index)
    return raw


def _to_numeric_score(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.str.lower().isin(INVALID_MARKERS | ABSENT_MARKERS | PREDICTABLE_MISSING_MARKERS), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _resolve_paper_count(row: pd.Series, metadata: list[PaperCountMetadata]) -> int | None:
    supplied_count = pd.to_numeric(pd.Series([row.get("paper_count")]), errors="coerce").iloc[0]
    if pd.notna(supplied_count) and int(supplied_count) in {2, 3, 4}:
        return int(supplied_count)
    code = None if pd.isna(row.get("subject_code")) else str(row.get("subject_code")).strip()
    if code and code.lower() not in METADATA_MISSING_MARKERS:
        last_digit = code[-1]
        if last_digit in {"2", "3", "4"}:
            return int(last_digit)
    subject_name = str(row.get("subject_name", "")).strip().lower()
    subject_id = str(row.get("subject_id", "")).strip()
    for item in metadata:
        if item.subject_id and subject_id and item.subject_id.strip() == subject_id:
            return item.paper_count
        if item.subject_code and code and str(item.subject_code).strip() == code:
            return item.paper_count
        if item.subject_name and item.subject_name.strip().lower() == subject_name:
            return item.paper_count
    return None


def _classify_score_records(data: pd.DataFrame, raw_score_columns: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    invalid_mask = pd.Series(False, index=data.index)
    absent_mask = pd.Series(False, index=data.index)
    invalid_reasons = pd.Series("", index=data.index, dtype="object")
    absent_reasons = pd.Series("", index=data.index, dtype="object")

    for paper_idx, score_col in enumerate(PAPER_SCORE_COLUMNS, start=1):
        applicable = data["paper_count"] >= paper_idx
        raw = raw_score_columns[score_col].astype(str).str.strip().str.lower()
        paper_label = f"P{paper_idx}"
        paper_invalid = applicable & raw.isin(INVALID_MARKERS)
        paper_absent = applicable & raw.isin(ABSENT_MARKERS)
        invalid_mask |= paper_invalid
        absent_mask |= paper_absent
        invalid_reasons.loc[paper_invalid] = invalid_reasons.loc[paper_invalid].map(
            lambda value: f"{value}; {paper_label} invalid value".strip("; ")
        )
        absent_reasons.loc[paper_absent] = absent_reasons.loc[paper_absent].map(
            lambda value: f"{value}; {paper_label} absent".strip("; ")
        )

    invalid_records = data.loc[invalid_mask].copy()
    if not invalid_records.empty:
        invalid_records["record_status"] = "invalid"
        invalid_records["record_reason"] = invalid_reasons.loc[invalid_mask].values

    absent_records = data.loc[~invalid_mask & absent_mask].copy()
    if not absent_records.empty:
        absent_records["record_status"] = "absent"
        absent_records["record_reason"] = absent_reasons.loc[~invalid_mask & absent_mask].values

    cleanable_data = data.loc[~invalid_mask & ~absent_mask].copy()
    return {
        "cleanable_data": cleanable_data,
        "invalid_records": invalid_records,
        "absent_records": absent_records,
    }


def _drop_non_applicable_scores(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for paper_idx in range(1, 5):
        inactive = output["paper_count"] < paper_idx
        output.loc[inactive, f"p{paper_idx}_score"] = np.nan
        output.loc[inactive, f"p{paper_idx}_max"] = np.nan
    return output


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
        if item.subject_id:
            mask &= data["subject_id"].astype(str).str.strip() == item.subject_id.strip()
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

    applicable_scores = output[PAPER_SCORE_COLUMNS]
    output["partial_total"] = applicable_scores.sum(axis=1, skipna=True)
    output["mean_score"] = applicable_scores.mean(axis=1, skipna=True)
    output["score_spread"] = applicable_scores.max(axis=1, skipna=True) - applicable_scores.min(axis=1, skipna=True)
    output["score_std"] = applicable_scores.std(axis=1, skipna=True).fillna(0)
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
        "subject_id",
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


def _canonical_order_if_possible(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    columns = [column for column in _canonical_order(data).columns if column in data.columns]
    return data[columns]


def _append_records(existing: pd.DataFrame, new_records: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new_records.copy()
    if new_records.empty:
        return existing
    return pd.concat([existing, new_records], ignore_index=True, sort=False)
