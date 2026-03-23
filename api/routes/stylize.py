from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import Prediction

router = APIRouter(tags=["stylize"])

VALID_STYLES = {"vangogh", "bw", "hokusai", "cartoon"}


class ColorizeRequest(BaseModel):
    image: str                       # base64-encoded image (grayscale or RGB)
    model_path: str | None = None    # optional: path to a specific colorizer .h5


class ColorizerInfo(BaseModel):
    path: str
    name: str
    dataset: str


class StyleRequest(BaseModel):
    image: str  # base64-encoded image
    style: str  # one of VALID_STYLES


class ImageResponse(BaseModel):
    result_image: str  # base64-encoded JPEG result


@router.get("/colorize/models", response_model=list[ColorizerInfo])
async def list_colorizer_models() -> list[ColorizerInfo]:
    """Return all available colorizer .h5 models."""
    from ml.image_model.predict import list_colorizers
    return [ColorizerInfo(**m) for m in list_colorizers()]


@router.post("/colorize", response_model=ImageResponse)
async def colorize_image(
    request: ColorizeRequest,
    db: AsyncSession = Depends(get_db),
) -> ImageResponse:
    try:
        img_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        from ml.image_model.predict import colorize
        result_bytes = colorize(img_bytes, model_path=request.model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    db.add(Prediction(model_type="colorizer"))

    return ImageResponse(result_image=base64.b64encode(result_bytes).decode())


@router.post("/stylize", response_model=ImageResponse)
async def stylize_image(request: StyleRequest) -> ImageResponse:
    if request.style not in VALID_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"style must be one of {sorted(VALID_STYLES)}",
        )

    try:
        img_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    from ml.image_model.predict import apply_style

    result_bytes = apply_style(img_bytes, request.style)
    return ImageResponse(result_image=base64.b64encode(result_bytes).decode())
