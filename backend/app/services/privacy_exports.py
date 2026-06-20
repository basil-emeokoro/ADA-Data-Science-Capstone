from __future__ import annotations

from pathlib import Path

import pandas as pd


PUBLIC_EXPORT_COLUMNS = [
    "subject_id",
    "paper_count",
    "anonymized_candidate_id",
    "p1_score",
    "p2_score",
    "p3_score",
    "p4_score",
    "p1_max",
    "p2_max",
    "p3_max",
    "p4_max",
]

PRIVATE_MAPPING_COLUMNS = ["original_subject_code", "public_subject_id", "subject_name", "paper_count"]


def build_public_research_dataset(
    data: pd.DataFrame,
    mapping_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a public dataset and private subject mapping without leaking subject codes."""

    mapping = _load_or_extend_subject_mapping(data, mapping_path)
    output = data.copy()
    output["subject_id"] = output.apply(lambda row: _lookup_public_subject_id(row, mapping), axis=1)
    for column in PUBLIC_EXPORT_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    public = output[PUBLIC_EXPORT_COLUMNS].copy()
    return public, mapping


def private_subject_mapping_path(project_root: Path) -> Path:
    return project_root / "Working Documents" / "subject_mapping_private.csv"


def _load_or_extend_subject_mapping(data: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    existing = _read_mapping(mapping_path)
    mapping_records = existing.to_dict("records")
    known_codes = {str(row["original_subject_code"]).strip() for row in mapping_records if str(row.get("original_subject_code", "")).strip()}

    subjects = _subject_records(data)
    next_index = _next_subject_index(existing)
    for subject in subjects:
        code = subject["original_subject_code"]
        if code in known_codes:
            continue
        subject["public_subject_id"] = f"SUBJ_{next_index:03d}"
        mapping_records.append(subject)
        known_codes.add(code)
        next_index += 1

    mapping = pd.DataFrame(mapping_records, columns=PRIVATE_MAPPING_COLUMNS)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_path, index=False)
    return mapping


def _read_mapping(mapping_path: Path) -> pd.DataFrame:
    if not mapping_path.exists() or mapping_path.stat().st_size == 0:
        return pd.DataFrame(columns=PRIVATE_MAPPING_COLUMNS)
    mapping = pd.read_csv(mapping_path, dtype={"original_subject_code": str, "public_subject_id": str, "subject_name": str})
    for column in PRIVATE_MAPPING_COLUMNS:
        if column not in mapping.columns:
            mapping[column] = pd.NA
    return mapping[PRIVATE_MAPPING_COLUMNS]


def _subject_records(data: pd.DataFrame) -> list[dict[str, object]]:
    if "subject_code" not in data.columns:
        return []
    frame = data.copy()
    frame["original_subject_code"] = frame["subject_code"].astype(str).str.strip()
    frame = frame[frame["original_subject_code"].notna() & (frame["original_subject_code"] != "") & (frame["original_subject_code"].str.lower() != "nan")]
    if frame.empty:
        return []
    if "subject_name" not in frame.columns:
        frame["subject_name"] = ""
    if "paper_count" not in frame.columns:
        frame["paper_count"] = pd.NA
    grouped = (
        frame.groupby("original_subject_code", dropna=False)
        .agg({"subject_name": "first", "paper_count": "first"})
        .reset_index()
        .sort_values("original_subject_code", kind="stable")
    )
    records: list[dict[str, object]] = []
    for _, row in grouped.iterrows():
        records.append(
            {
                "original_subject_code": str(row["original_subject_code"]).strip(),
                "public_subject_id": "",
                "subject_name": "" if pd.isna(row["subject_name"]) else str(row["subject_name"]).strip(),
                "paper_count": None if pd.isna(row["paper_count"]) else int(row["paper_count"]),
            }
        )
    return records


def _next_subject_index(mapping: pd.DataFrame) -> int:
    if mapping.empty or "public_subject_id" not in mapping.columns:
        return 1
    indexes = []
    for value in mapping["public_subject_id"].dropna().astype(str):
        if value.startswith("SUBJ_") and value[5:].isdigit():
            indexes.append(int(value[5:]))
    return max(indexes, default=0) + 1


def _lookup_public_subject_id(row: pd.Series, mapping: pd.DataFrame) -> str | None:
    code = str(row.get("subject_code", "")).strip()
    if not code or code.lower() == "nan":
        return None
    match = mapping[mapping["original_subject_code"].astype(str).str.strip() == code]
    if match.empty:
        return None
    return str(match.iloc[0]["public_subject_id"])
