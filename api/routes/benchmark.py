from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models import BenchmarkRun

logger = logging.getLogger(__name__)
router = APIRouter(tags=["benchmark"])


class BenchmarkRequest(BaseModel):
    dataset: str = "breast_cancer"
    epochs: int = 100


@router.post("/benchmark/run")
async def run_benchmark_endpoint(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    background_tasks.add_task(_run_and_save, request.dataset, request.epochs)
    return {"status": "started", "dataset": request.dataset}


@router.get("/benchmark/results")
async def get_results(db: AsyncSession = Depends(get_db)) -> list[dict]:
    stmt = select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(10)
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "dataset": r.dataset,
            "results": r.results,
            "chart_path": r.chart_path,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


async def _run_and_save(dataset: str, epochs: int) -> None:
    try:
        from ml.benchmark.run_benchmark import run_benchmark

        result = await asyncio.to_thread(run_benchmark, dataset=dataset, epochs=epochs)

        from api.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            session.add(
                BenchmarkRun(
                    dataset=dataset,
                    results=result["results"],
                    chart_path=result.get("chart_path"),
                )
            )
            await session.commit()
    except Exception:
        logger.exception("Benchmark task failed")
