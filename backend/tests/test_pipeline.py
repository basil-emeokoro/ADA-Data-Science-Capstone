from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.ml.preprocessing.cleaning import clean_dataset
from backend.app.schemas.pipeline import MaxScoreMetadata, PaperCountMetadata, ProcessingRequest, PredictionMode
from backend.app.services.anonymization import anonymize_dataset, detect_sensitive_fields
from backend.app.services.csv_parser import parse_examination_csv
from backend.app.services.pipeline import run_pipeline


def _csv(rows: list[tuple[int, int, int]], subject_code: str = "718003") -> bytes:
    lines = [
        ",,,,,,",
        "S/N,Subject Code,Center Number,Candidate Number,Paper 1,Paper 2,Paper 3",
        ",,,,Max:40,Max:60,Max:100",
    ]
    for idx, (p1, p2, p3) in enumerate(rows, start=1):
        lines.append(f"{idx},{subject_code},4380102,{idx:03d},{p1},{p2},{p3}")
    return ("\n".join(lines) + "\n").encode()


def _rows(n: int = 18) -> list[tuple[int, int, int]]:
    return [(20 + i % 15, 31 + (i * 2) % 25, 55 + (i * 3) % 35) for i in range(n)]


def test_parse_sample_style_csv_and_detect_maxima() -> None:
    df, maxima = parse_examination_csv(_csv(_rows(8)))
    assert list(df.columns) == ["serial_no", "subject_code", "centre_no", "candidate_number", "p1_score", "p2_score", "p3_score"]
    assert maxima == {"p1_max": 40.0, "p2_max": 60.0, "p3_max": 100.0}
    assert len(df) == 8


def test_anonymization_replaces_candidate_identifiers() -> None:
    df, _ = parse_examination_csv(_csv(_rows(3)))
    anonymized = anonymize_dataset(df)
    assert "anonymized_candidate_id" in anonymized.columns
    assert "centre_no" not in anonymized.columns
    assert "candidate_number" not in anonymized.columns
    assert anonymized["anonymized_candidate_id"].iloc[0] == "CAND_000001"


def test_missing_metadata_requires_paper_count() -> None:
    data = pd.DataFrame({"subject_name": ["Biology"], "p1_score": [20], "p2_score": [30], "p1_max": [40], "p2_max": [60]})
    result = clean_dataset(data)
    assert result.errors
    recovered = clean_dataset(data, paper_counts=[PaperCountMetadata(subject_name="Biology", paper_count=2)])
    assert not recovered.errors
    assert recovered.data["paper_count"].iloc[0] == 2


def test_missing_maxima_requires_recovery() -> None:
    data = pd.DataFrame({"subject_code": ["302002"], "p1_score": [20], "p2_score": [30]})
    result = clean_dataset(data)
    assert result.errors
    recovered = clean_dataset(data, max_scores=[MaxScoreMetadata(subject_code="302002", p1_max=40, p2_max=60)])
    assert not recovered.errors


def test_invalid_score_above_maximum_is_rejected() -> None:
    data = pd.DataFrame({"subject_code": ["302002"], "p1_score": [57], "p2_score": [30], "p1_max": [40], "p2_max": [60]})
    result = clean_dataset(data)
    assert result.errors
    assert "exceed" in result.errors[0]


def test_mode_a_runs_benchmark_and_exports() -> None:
    request = ProcessingRequest(mode=PredictionMode.mode_a)
    response = run_pipeline(_csv(_rows(18)), "sample.csv", request)
    assert not response.errors
    assert response.exports["ada_safe_dataset"]
    assert response.exports["metrics"]
    assert response.metrics
    assert response.rankings
    assert "actual_vs_predicted" in response.plots
    assert "shap" in response.plots


def test_process_api_serializes_plotly_outputs() -> None:
    client = TestClient(app)
    request = ProcessingRequest(mode=PredictionMode.mode_a)
    response = client.post(
        "/api/process",
        data={"payload": request.model_dump_json()},
        files={"file": ("sample.csv", _csv(_rows(18)), "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]
    assert "actual_vs_predicted" in payload["plots"]


def test_mode_b_predicts_single_missing_score_and_exports() -> None:
    rows = _rows(18)
    lines = [
        "S/N,Subject Code,Center Number,Candidate Number,Paper 1,Paper 2,Paper 3",
        ",,,,Max:40,Max:60,Max:100",
    ]
    for idx, (p1, p2, p3) in enumerate(rows, start=1):
        p3_value = "missing" if idx == 18 else p3
        lines.append(f"{idx},718003,4380102,{idx:03d},{p1},{p2},{p3_value}")
    request = ProcessingRequest(mode=PredictionMode.mode_b)
    response = run_pipeline(("\n".join(lines) + "\n").encode(), "mode_b.csv", request)
    assert not response.errors
    assert response.exports["completed_prediction_file"]
    exported = pd.read_csv(response.exports["completed_prediction_file"])
    assert "prediction_status" in exported.columns
    assert "predicted" in set(exported["prediction_status"])


def test_mode_b_rejects_multiple_missing_papers_with_warning() -> None:
    rows = _rows(18)
    lines = [
        "S/N,Subject Code,Center Number,Candidate Number,Paper 1,Paper 2,Paper 3",
        ",,,,Max:40,Max:60,Max:100",
    ]
    for idx, (p1, p2, p3) in enumerate(rows, start=1):
        p1_value = "missing" if idx == 18 else p1
        p2_value = "missing" if idx == 18 else p2
        lines.append(f"{idx},718003,4380102,{idx:03d},{p1_value},{p2_value},{p3}")
    response = run_pipeline(("\n".join(lines) + "\n").encode(), "invalid_mode_b.csv", ProcessingRequest(mode=PredictionMode.mode_b))
    assert any("multiple missing papers" in warning for warning in response.warnings)
    assert "unpredictable_reference_file" in response.exports
    reference = pd.read_csv(response.exports["unpredictable_reference_file"])
    assert "unpredictable_reason" in reference.columns
    assert "multiple missing paper scores" in set(reference["unpredictable_reason"])


def test_multi_subject_mode_a() -> None:
    first = [f"{idx},718003,4380102,{idx:03d},{20 + idx % 10},{35 + idx % 12},{60 + idx % 20}" for idx in range(1, 13)]
    second = [f"{idx + 20},302002,4380103,{idx:03d},{18 + idx % 10},{31 + idx % 15}," for idx in range(1, 13)]
    payload = "\n".join(
        [
            "S/N,Subject Code,Center Number,Candidate Number,Paper 1,Paper 2,Paper 3",
            ",,,,Max:40,Max:60,Max:100",
            *first,
            *second,
        ]
    )
    response = run_pipeline(payload.encode(), "multi.csv", ProcessingRequest(mode=PredictionMode.mode_a))
    assert not response.errors
    assert {"718003", "302002"}.issubset({row["subject"] for row in response.metrics})


def test_mode_a_filters_incomplete_records_before_benchmarking() -> None:
    rows = _rows(18)
    lines = [
        "S/N,Subject Code,Center Number,Candidate Number,Paper 1,Paper 2,Paper 3",
        ",,,,Max:40,Max:60,Max:100",
    ]
    for idx, (p1, p2, p3) in enumerate(rows, start=1):
        p2_value = "missing" if idx == 18 else p2
        lines.append(f"{idx},718003,4380102,{idx:03d},{p1},{p2_value},{p3}")
    response = run_pipeline(("\n".join(lines) + "\n").encode(), "mode_a_incomplete.csv", ProcessingRequest(mode=PredictionMode.mode_a))
    assert not response.errors
    assert response.rows == 17
    assert any("incomplete record" in warning for warning in response.warnings)
