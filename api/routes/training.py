from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.training_manager import training_manager

router = APIRouter(tags=["training"])


class TrainingStartRequest(BaseModel):
    preset: str


@router.get("/training/presets")
async def get_presets() -> dict:
    return {"presets": training_manager.list_presets()}


@router.get("/training/jobs")
async def get_jobs() -> dict:
    return {"jobs": training_manager.list_jobs()}


@router.get("/training/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.post("/training/start")
async def start_training(request: TrainingStartRequest) -> dict:
    try:
        return training_manager.start(request.preset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
