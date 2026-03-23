from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.database import init_db
from api.routes import benchmark, classify, dashboard, neural_style, stylize, training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database…")
    await init_db()
    logger.info("Database ready.")
    yield


app = FastAPI(title="ML Sandbox API", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="api/static"), name="static")

app.include_router(classify.router, prefix="/api")
app.include_router(stylize.router, prefix="/api")
app.include_router(neural_style.router, prefix="/api")
app.include_router(benchmark.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(training.router, prefix="/api")


@app.get("/")
async def root():
    return FileResponse("api/static/index.html")


@app.get("/training")
async def training_page():
    return FileResponse("api/static/training.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": _models_present(),
    }


def _models_present() -> bool:
    from api.config import settings

    return os.path.exists(settings.model_classifier) and os.path.exists(
        settings.model_colorizer
    )
