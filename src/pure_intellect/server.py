"""FastAPI сервер для Чистый Интеллект."""

import logging
import time
import importlib.metadata
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api.routes import router, openai_router
from .api.websocket import websocket_endpoint
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="Чистый Интеллект",
    description="Локальный оркестратор для LLM с иерархической памятью",
    version="0.1.0",
)

# CORS — разрешаем все origins для локального использования и Docker
_CORS_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1", tags=["orchestrator"])
app.include_router(openai_router, tags=["openai-compatible"])

# WebSocket
app.add_api_websocket_route("/ws", websocket_endpoint)

# Static files — Web UI
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root(response: Response):
    """Отдаём Web UI — без кэширования браузером."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/v1/version", include_in_schema=False)
async def version_info():
    """Версия и диагностика — для проверки что новая версия установлена."""
    try:
        ver = importlib.metadata.version("pure-intellect")
    except Exception:
        ver = "dev"
    index_path = _STATIC_DIR / "index.html"
    return {
        "version": ver,
        "static_dir": str(_STATIC_DIR),
        "index_html_exists": index_path.exists(),
        "index_html_size_bytes": index_path.stat().st_size if index_path.exists() else 0,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }



async def _preload_models():
    """Предзагрузка coordinator и generator в VRAM при старте сервера.
    
    Посылает минимальный запрос к каждой модели с keep_alive=-1
    чтобы Ollama держал их постоянно в памяти (как LM Studio).
    """
    import httpx
    from .engines.config_loader import load_config
    await asyncio.sleep(3)  # дать серверу полностью подняться
    try:
        cfg = load_config()
        coordinator = cfg.coordinator.model
        generator = cfg.generator.model
        logger.info(f"🔥 Preloading models: {coordinator} + {generator}")
        async with httpx.AsyncClient(timeout=300.0) as client:
            for model in [coordinator, generator]:
                try:
                    resp = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model, "prompt": "", "keep_alive": -1},
                    )
                    if resp.status_code == 200:
                        logger.info(f"   ✅ {model} loaded into VRAM")
                    else:
                        logger.warning(f"   ⚠️ {model} preload failed: {resp.status_code}")
                except Exception as e:
                    logger.warning(f"   ⚠️ {model} preload error: {e}")
    except Exception as e:
        logger.warning(f"Model preload failed: {e}")

@app.on_event("startup")
async def startup():
    """Инициализация при запуске."""
    logger.info("🧠 Чистый Интеллект запускается...")
    logger.info(f"   Orchestrator model: {settings.orchestrator_model}")
    logger.info(f"   Chat model: {settings.chat_model}")
    logger.info(f"   Storage: {settings.storage_dir}")
    logger.info(f"   Static dir: {_STATIC_DIR}")
    logger.info(f"   index.html: {'OK' if (_STATIC_DIR / 'index.html').exists() else 'MISSING!'}")
    # Предзагрузка обеих моделей в VRAM при старте
    import asyncio
    asyncio.ensure_future(_preload_models())


@app.on_event("shutdown")
async def shutdown():
    """Очистка при остановке — освобождаем VRAM."""
    from .engine.model_manager import ModelManager
    manager = ModelManager.get_instance()
    manager.dispose()
    logger.info("🧠 Чистый Интеллект остановлен, VRAM освобождена")
