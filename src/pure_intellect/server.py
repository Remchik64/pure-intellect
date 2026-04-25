"""FastAPI сервер для Чистый Интеллект."""

import asyncio
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
    """Умная предзагрузка моделей в VRAM с проверкой доступной памяти.
    
    Логика:
    1. Получаем размер каждой модели через /api/show
    2. Получаем доступный VRAM через /api/ps
    3. Если обе влезают -> загружаем обе с keep_alive=-1 (постоянно)
    4. Если не влезают -> загружаем только generator с keep_alive=-1,
       coordinator загружается по запросу (он маленький, быстро)
    """
    import httpx
    from .engines.config_loader import load_config
    await asyncio.sleep(3)  # дать серверу полностью подняться
    try:
        cfg = load_config()
        coordinator = cfg.coordinator.model
        generator = cfg.generator.model
        logger.info(f"🧠 VRAM check before preload...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Шаг 1: получаем размеры моделей
            sizes = {}
            for model in [coordinator, generator]:
                try:
                    resp = await client.post(
                        "http://localhost:11434/api/show",
                        json={"name": model}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        # size в байтах
                        size_bytes = data.get("size", 0)
                        sizes[model] = size_bytes
                        size_gb = size_bytes / (1024**3)
                        logger.info(f"   {model}: {size_gb:.1f} GB")
                    else:
                        logger.warning(f"   Cannot get size for {model}: {resp.status_code}")
                        sizes[model] = 0
                except Exception as e:
                    logger.warning(f"   Size check failed for {model}: {e}")
                    sizes[model] = 0

            # Шаг 2: получаем доступный VRAM
            available_vram_bytes = 0
            try:
                ps_resp = await client.get("http://localhost:11434/api/ps")
                if ps_resp.status_code == 200:
                    # Ollama не возвращает free VRAM напрямую
                    # Используем nvidia-smi через subprocess
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        free_mb = int(result.stdout.strip().split("\n")[0])
                        available_vram_bytes = free_mb * 1024 * 1024
                        logger.info(f"   Free VRAM: {free_mb / 1024:.1f} GB")
            except Exception as e:
                logger.warning(f"   VRAM check failed: {e}")
                # Assume 8GB available as safe default
                available_vram_bytes = 8 * 1024**3

            # Шаг 3: решаем стратегию загрузки
            total_needed = sum(sizes.values())
            # Добавляем 10% запас для KV cache и compute
            total_needed_with_overhead = total_needed * 1.1
            
            coordinator_gb = sizes.get(coordinator, 0) / (1024**3)
            generator_gb = sizes.get(generator, 0) / (1024**3)
            total_gb = total_needed_with_overhead / (1024**3)
            free_gb = available_vram_bytes / (1024**3)
            
            logger.info(f"   Need: {total_gb:.1f} GB | Free: {free_gb:.1f} GB")

            if total_needed_with_overhead <= available_vram_bytes:
                # Обе модели влезают -> загружаем обе постоянно
                logger.info(f"🟢 Both models fit! Loading both with keep_alive=-1")
                for model in [coordinator, generator]:
                    try:
                        resp = await client.post(
                            "http://localhost:11434/api/generate",
                            json={"model": model, "prompt": "", "keep_alive": -1},
                        )
                        if resp.status_code == 200:
                            logger.info(f"   ✅ {model} loaded (permanent)")
                        else:
                            logger.warning(f"   ⚠️ {model}: {resp.status_code}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ {model}: {e}")
            else:
                # Не влезают обе -> загружаем только generator постоянно
                # Coordinator маленький (2-3B), загрузится быстро по запросу
                logger.info(f"🟡 Not enough VRAM for both. Loading generator only.")
                logger.info(f"   Coordinator ({coordinator_gb:.1f}GB) will load on-demand")
                try:
                    resp = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": generator, "prompt": "", "keep_alive": -1},
                    )
                    if resp.status_code == 200:
                        logger.info(f"   ✅ {generator} loaded (permanent)")
                    else:
                        logger.warning(f"   ⚠️ {generator}: {resp.status_code}")
                except Exception as e:
                    logger.warning(f"   ⚠️ {generator}: {e}")

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
    asyncio.ensure_future(_preload_models())


@app.on_event("shutdown")
async def shutdown():
    """Очистка при остановке — освобождаем VRAM."""
    from .engine.model_manager import ModelManager
    manager = ModelManager.get_instance()
    manager.dispose()
    logger.info("🧠 Чистый Интеллект остановлен, VRAM освобождена")
