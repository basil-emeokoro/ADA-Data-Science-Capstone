from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PredictionMode(str, Enum):
    mode_a = "mode_a"
    mode_b = "mode_b"


class PaperCountMetadata(BaseModel):
    subject_name: str | None = None
    subject_code: str | None = None
    paper_count: int = Field(ge=2, le=4)


class MaxScoreMetadata(BaseModel):
    subject_name: str | None = None
    subject_code: str | None = None
    p1_max: float | None = None
    p2_max: float | None = None
    p3_max: float | None = None
    p4_max: float | None = None


class ProcessingRequest(BaseModel):
    mode: PredictionMode
    paper_counts: list[PaperCountMetadata] = Field(default_factory=list)
    max_scores: list[MaxScoreMetadata] = Field(default_factory=list)


class ProcessingResponse(BaseModel):
    mode: PredictionMode
    rows: int
    exports: dict[str, str] = Field(default_factory=dict)
    export_downloads: dict[str, str] = Field(default_factory=dict)
    metrics: list[dict[str, Any]] = Field(default_factory=list)
    rankings: list[dict[str, Any]] = Field(default_factory=list)
    plots: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SubjectDetection(BaseModel):
    subject_key: str
    subject_code: str | None = None
    subject_name: str | None = None
    inferred_paper_count: int | None = None
    row_count: int
    detected_max_scores: dict[str, float | None] = Field(default_factory=dict)


class DetectionResponse(BaseModel):
    filename: str
    columns: list[str]
    sensitive_fields: list[str]
    inferred_paper_count: int | None
    detected_max_scores: dict[str, float | None]
    row_count: int
    subjects: list[SubjectDetection] = Field(default_factory=list)


class AdaSafeExportResponse(BaseModel):
    rows: int
    export_path: str
    download_url: str
    sensitive_fields: list[str]
    columns: list[str]
