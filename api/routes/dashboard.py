from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import get_db
from api.models import BenchmarkRun, BotActivity, Prediction, User

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/overview")
async def dashboard_overview(db: AsyncSession = Depends(get_db)) -> dict:
    prediction_total = await db.scalar(select(func.count(Prediction.id)))
    activity_total = await db.scalar(select(func.count(BotActivity.id)))
    benchmark_total = await db.scalar(select(func.count(BenchmarkRun.id)))
    user_total = await db.scalar(select(func.count(User.id)))

    latest_predictions = (
        await db.execute(
            select(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(8)
        )
    ).scalars().all()

    latest_benchmarks = (
        await db.execute(
            select(BenchmarkRun)
            .order_by(BenchmarkRun.created_at.desc())
            .limit(5)
        )
    ).scalars().all()

    latest_activity = (
        await db.execute(
            select(BotActivity, User.username, User.telegram_id)
            .join(User, BotActivity.user_id == User.id, isouter=True)
            .order_by(BotActivity.timestamp.desc())
            .limit(8)
        )
    ).all()

    return {
        "health": {
            "status": "ok",
            "models_loaded": _models_present(),
            "classifier_path": settings.model_classifier,
            "colorizer_path": settings.model_colorizer,
        },
        "totals": {
            "predictions": prediction_total or 0,
            "activity": activity_total or 0,
            "benchmarks": benchmark_total or 0,
            "users": user_total or 0,
        },
        "recent_predictions": [
            {
                "id": row.id,
                "model_type": row.model_type,
                "label": row.label,
                "confidence": row.confidence,
                "created_at": row.created_at.isoformat(),
            }
            for row in latest_predictions
        ],
        "recent_activity": [
            {
                "id": activity.id,
                "username": username,
                "telegram_id": telegram_id,
                "command": activity.command,
                "timestamp": activity.timestamp.isoformat(),
            }
            for activity, username, telegram_id in latest_activity
        ],
        "recent_benchmarks": [
            {
                "id": row.id,
                "dataset": row.dataset,
                "created_at": row.created_at.isoformat(),
                "results": row.results.get("results", []),
            }
            for row in latest_benchmarks
        ],
    }


def _models_present() -> bool:
    return os.path.exists(settings.model_classifier) and os.path.exists(
        settings.model_colorizer
    )
