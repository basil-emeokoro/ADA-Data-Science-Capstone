from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.config.settings import get_settings
from backend.app.schemas.pipeline import (
    AdaSafeExportResponse,
    CleaningPreviewResponse,
    ColumnMappingRequest,
    DetectionResponse,
    ProcessingRequest,
    ProcessingResponse,
)
from backend.app.services.pipeline import detect_upload, export_ada_safe_dataset, preview_cleaning, run_pipeline


router = APIRouter(prefix="/api", tags=["capstone"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/download/{filename}")
def download_export(filename: str) -> FileResponse:
    export_dir = get_settings().export_dir.resolve()
    path = (export_dir / filename).resolve()
    if path.parent != export_dir or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Export file not found.")
    return FileResponse(path, filename=path.name)


@router.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...), payload: str | None = Form(None)) -> DetectionResponse:
    content = await file.read()
    try:
        mapping = ColumnMappingRequest.model_validate_json(payload).column_mapping if payload else {}
        return DetectionResponse(**detect_upload(content, file.filename or "upload.csv", mapping))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/export/ada-safe", response_model=AdaSafeExportResponse)
async def export_ada_safe(file: UploadFile = File(...), payload: str | None = Form(None)) -> AdaSafeExportResponse:
    content = await file.read()
    try:
        request = ProcessingRequest.model_validate_json(payload) if payload else None
        return export_ada_safe_dataset(content, file.filename or "upload.csv", request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/cleaning/preview", response_model=CleaningPreviewResponse)
async def cleaning_preview(file: UploadFile = File(...), payload: str = Form(...)) -> CleaningPreviewResponse:
    content = await file.read()
    try:
        request = ProcessingRequest.model_validate_json(payload)
        return preview_cleaning(content, file.filename or "upload.csv", request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/process", response_model=ProcessingResponse)
async def process(file: UploadFile = File(...), payload: str = Form(...)) -> ProcessingResponse:
    content = await file.read()
    try:
        request = ProcessingRequest.model_validate_json(payload)
        return run_pipeline(content, file.filename or "upload.csv", request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
