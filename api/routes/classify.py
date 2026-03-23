from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import Prediction

router = APIRouter(tags=["classify"])


class ClassifyRequest(BaseModel):
    image: str           # base64-encoded image bytes
    model_path: str | None = None  # optional: path to a specific classifier .h5


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]


class ClassifierInfo(BaseModel):
    path: str
    name: str
    dataset: str
    num_classes: int | None


@router.get("/classify/models", response_model=list[ClassifierInfo])
async def list_classifier_models() -> list[ClassifierInfo]:
    """Return all available classifier .h5 models found in the models directory."""
    from ml.image_model.predict import list_classifiers
    return [ClassifierInfo(**m) for m in list_classifiers()]


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(
    request: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
) -> ClassifyResponse:
    try:
        img_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        from ml.image_model.predict import classify
        result = classify(img_bytes, model_path=request.model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    db.add(Prediction(
        model_type="classifier",
        label=result["label"],
        confidence=result["confidence"],
    ))

    return ClassifyResponse(**result)


@router.post("/classify/imagenet", response_model=ClassifyResponse)
async def classify_imagenet(
    request: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
) -> ClassifyResponse:
    try:
        img_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    from ml.image_model.predict import classify_imagenet as _classify_imagenet
    result = _classify_imagenet(img_bytes)

    db.add(Prediction(
        model_type="imagenet",
        label=result["label"],
        confidence=result["confidence"],
    ))

    return ClassifyResponse(**result)
