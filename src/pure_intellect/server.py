"""FastAPI сервер для Чистый Интеллект."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Чистый Интеллект",
    description="Локальный оркестратор для LLM с иерархической памятью",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1", tags=["orchestrator"])


@app.on_event("startup")
async def startup():
    """Инициализация при запуске."""
    logger.info("🧠 Чистый Интеллект запускается...")
    logger.info(f"   Orchestrator model: {settings.orchestrator_model}")
    logger.info(f"   Chat model: {settings.chat_model}")
    logger.info(f"   Storage: {settings.storage_dir}")


@app.on_event("shutdown")
async def shutdown():
    """Очистка при остановке."""
    logger.info("🧠 Чистый Интеллект останавливается...")
