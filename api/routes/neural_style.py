from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.models import Prediction

router = APIRouter(tags=["neural-style"])


class NeuralStyleRequest(BaseModel):
    image: str        # base64-encoded content image
    style: str        # one of STYLE_PRESETS keys
    iterations: int = 200
    img_size: int = 384


class ImageResponse(BaseModel):
    result_image: str  # base64-encoded JPEG


@router.post("/neural-stylize", response_model=ImageResponse)
async def neural_stylize_image(request: NeuralStyleRequest) -> ImageResponse:
    """
    Apply Neural Style Transfer (VGG19, Gatys et al.) to the content image.

    Processing time: 5–15 s on GPU, up to ~60 s on CPU.
    Style images must be downloaded first: python -m ml.image_model.styles.download_styles
    """
    try:
        img_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    from ml.image_model.neural_style import STYLE_PRESETS, neural_stylize

    if request.style not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown style '{request.style}'. Available: {sorted(STYLE_PRESETS)}",
        )

    try:
        result_bytes = neural_stylize(
            img_bytes,
            style_name=request.style,
            iterations=request.iterations,
            img_size=request.img_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    return ImageResponse(result_image=base64.b64encode(result_bytes).decode())
