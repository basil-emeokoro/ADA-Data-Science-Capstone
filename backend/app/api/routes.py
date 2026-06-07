from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.schemas.pipeline import AdaSafeExportResponse, DetectionResponse, ProcessingRequest, ProcessingResponse
from backend.app.services.pipeline import detect_upload, export_ada_safe_dataset, run_pipeline


router = APIRouter(prefix="/api", tags=["capstone"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    content = await file.read()
    try:
        return DetectionResponse(**detect_upload(content, file.filename or "upload.csv"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/export/ada-safe", response_model=AdaSafeExportResponse)
async def export_ada_safe(file: UploadFile = File(...)) -> AdaSafeExportResponse:
    content = await file.read()
    try:
        return export_ada_safe_dataset(content, file.filename or "upload.csv")
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
