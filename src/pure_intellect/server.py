"""FastAPI сервер для Чистый Интеллект."""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api.routes import router
from .api.websocket import websocket_endpoint
from .config import get_settings
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Чистый Интеллект",
    description="Локальный оркестратор для LLM с иерархической памятью",
    version="0.1.0",
)

# CORS — разрешаем localhost для локального использования
_CORS_ORIGINS = [
    "http://localhost:3005",
    "http://localhost:5006",
    "http://localhost:8085",
    "http://127.0.0.1:3005",
    "http://127.0.0.1:5006",
    "http://127.0.0.1:8085",
    "http://localhost",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1", tags=["orchestrator"])

# WebSocket
app.add_api_websocket_route("/ws", websocket_endpoint)

# Static files — Web UI
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Отдаём Web UI чата."""
    return FileResponse(_STATIC_DIR / "index.html")




@app.on_event("startup")
async def startup():
    """Инициализация при запуске."""
    logger.info("🧠 Чистый Интеллект запускается...")
    logger.info(f"   Orchestrator model: {settings.orchestrator_model}")
    logger.info(f"   Chat model: {settings.chat_model}")
    logger.info(f"   Storage: {settings.storage_dir}")


@app.on_event("shutdown")
async def shutdown():
    """Очистка при остановке — освобождаем VRAM."""
    from .engine.model_manager import ModelManager
    manager = ModelManager.get_instance()
    manager.dispose()
    logger.info("🧠 Чистый Интеллект остановлен, VRAM освобождена")
