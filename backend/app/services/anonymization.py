from __future__ import annotations

import pandas as pd


SENSITIVE_KEYWORDS = ("candidate", "centre", "center", "identifier", "id")


def detect_sensitive_fields(df: pd.DataFrame) -> list[str]:
    fields: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if any(keyword in lowered for keyword in SENSITIVE_KEYWORDS):
            fields.append(column)
    return fields


def anonymize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Replace direct identifiers with irreversible synthetic candidate IDs."""

    output = df.copy()
    if "candidate_id" not in output.columns:
        parts = []
        for column in ("centre_no", "candidate_number"):
            if column in output.columns:
                parts.append(output[column].astype(str).str.strip())
        if parts:
            output["candidate_id"] = parts[0]
            for part in parts[1:]:
                output["candidate_id"] = output["candidate_id"] + part
    unique_ids = output.get("candidate_id", pd.Series(range(len(output)), index=output.index)).astype(str)
    mapping = {value: f"CAND_{idx:06d}" for idx, value in enumerate(pd.unique(unique_ids), start=1)}
    output["anonymized_candidate_id"] = unique_ids.map(mapping)
    for column in detect_sensitive_fields(output):
        if column != "anonymized_candidate_id":
            output = output.drop(columns=[column])
    return output
