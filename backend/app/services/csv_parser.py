from __future__ import annotations

import re
from io import BytesIO
from typing import Any

import pandas as pd


MAX_RE = re.compile(r"max\s*:?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _read_raw(upload_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(upload_bytes), header=None, dtype=str, keep_default_na=False)


def _find_header_row(raw: pd.DataFrame) -> int:
    header_keywords = {"candidate", "paper", "subject", "center", "centre"}
    best_index = 0
    best_score = -1
    for idx, row in raw.iterrows():
        values = [str(v).strip().lower() for v in row.tolist()]
        score = sum(any(keyword in value for keyword in header_keywords) for value in values)
        if score > best_score:
            best_score = score
            best_index = int(idx)
    return best_index


def parse_examination_csv(upload_bytes: bytes) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Parse flexible examination CSVs, including maxima rows such as Max:40."""

    raw = _read_raw(upload_bytes)
    header_idx = _find_header_row(raw)
    headers = [str(v).strip() or f"unnamed_{i}" for i, v in enumerate(raw.iloc[header_idx].tolist())]
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = headers
    df = df.dropna(how="all")
    df = df.loc[~(df.astype(str).apply(lambda r: "".join(r).strip(), axis=1) == "")]

    detected_max: dict[str, float | None] = {}
    rows_to_drop: list[int] = []
    for idx, row in df.iterrows():
        row_text = " ".join(str(v) for v in row.tolist())
        if "max" in row_text.lower():
            rows_to_drop.append(idx)
            for col, value in row.items():
                match = MAX_RE.search(str(value))
                if match:
                    normalized = normalize_column_name(col)
                    if normalized in {"p1_score", "p2_score", "p3_score", "p4_score"}:
                        detected_max[normalized.replace("_score", "_max")] = float(match.group(1))
    if rows_to_drop:
        df = df.drop(index=rows_to_drop)

    df = df.reset_index(drop=True)
    df.columns = [normalize_column_name(col) for col in df.columns]
    for max_col in ("p1_max", "p2_max", "p3_max", "p4_max"):
        if max_col not in df.columns:
            continue
        values = pd.to_numeric(df[max_col], errors="coerce").dropna().unique()
        if len(values) == 1:
            detected_max[max_col] = float(values[0])
    return df, detected_max


def normalize_column_name(column: Any) -> str:
    name = str(column).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    aliases = {
        "s_n": "serial_no",
        "sn": "serial_no",
        "subject_code": "subject_code",
        "subject_id": "subject_id",
        "subjcode": "subject_code",
        "subj_code": "subject_code",
        "subject": "subject_name",
        "subject_name": "subject_name",
        "center_number": "centre_no",
        "centre_number": "centre_no",
        "center_no": "centre_no",
        "centre_no": "centre_no",
        "candidate_number": "candidate_number",
        "candidate_no": "candidate_number",
        "candidate_no": "candidate_number",
        "candidate": "candidate_number",
        "candidate_id": "candidate_id",
        "anonymized_candidate_id": "anonymized_candidate_id",
        "paper_count": "paper_count",
        "paper_1": "p1_score",
        "paper_2": "p2_score",
        "paper_3": "p3_score",
        "paper_4": "p4_score",
        "p1": "p1_score",
        "p2": "p2_score",
        "p3": "p3_score",
        "p4": "p4_score",
        "p1_score": "p1_score",
        "p2_score": "p2_score",
        "p3_score": "p3_score",
        "p4_score": "p4_score",
        "p1_max": "p1_max",
        "p2_max": "p2_max",
        "p3_max": "p3_max",
        "p4_max": "p4_max",
    }
    return aliases.get(name, name)
