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



async def _load_az_plugin_utility_model() -> str:
    """Читает utility_model из az_plugin_config.yaml."""
    import yaml, pathlib
    for candidate in [
        pathlib.Path("az_plugin_config.yaml"),
        pathlib.Path.home() / "AppData/Roaming/pure_intellect/az_plugin_config.yaml",
        pathlib.Path.home() / ".pure_intellect/az_plugin_config.yaml",
        pathlib.Path("/a0/usr/workdir/pure-intellect/az_plugin_config.yaml"),
    ]:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    cfg = yaml.safe_load(f) or {}
                return cfg.get("utility_model", "")
            except Exception:
                pass
    return ""


async def _preload_models():
    """Умная предзагрузка моделей в VRAM с проверкой доступной памяти.
    
    Логика:
    1. Получаем размер каждой модели через /api/show
    2. Получаем доступный VRAM через nvidia-smi или Ollama /api/ps
    3. Если все три влезают -> загружаем все три с keep_alive=-1
    4. Если две влезают -> coordinator + generator
    5. Если одна -> только generator
    """
    import httpx
    from .engines.config_loader import load_config
    await asyncio.sleep(3)  # дать серверу полностью подняться
    try:
        cfg = load_config()
        coordinator = cfg.coordinator.model
        generator = cfg.generator.model
        utility = await _load_az_plugin_utility_model()

        models_to_load = [coordinator, generator]
        if utility and utility not in models_to_load:
            models_to_load.append(utility)
            logger.info(f"   Utility model: {utility}")

        logger.info(f"🧠 VRAM check before preload...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Шаг 1: получаем размеры моделей
            sizes = {}
            for model in models_to_load:
                try:
                    resp = await client.post(
                        "http://localhost:11434/api/show",
                        json={"name": model}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
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
                import subprocess, sys
                smi_paths = ["nvidia-smi"]
                if sys.platform == "win32":
                    smi_paths += [
                        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                        r"C:\Windows\System32\nvidia-smi.exe",
                    ]
                for smi in smi_paths:
                    try:
                        result = subprocess.run(
                            [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            free_mb = int(result.stdout.strip().split("\n")[0])
                            available_vram_bytes = free_mb * 1024 * 1024
                            logger.info(f"   Free VRAM: {free_mb / 1024:.1f} GB")
                            break
                    except (FileNotFoundError, ValueError):
                        continue
                # Fallback: через Ollama /api/tags — берём total VRAM из первой запущенной модели
                if available_vram_bytes == 0:
                    try:
                        ps = await client.get("http://localhost:11434/api/ps")
                        if ps.status_code == 200:
                            ps_data = ps.json()
                            running = ps_data.get("models", [])
                            if running:
                                # total size_vram уже занято — считаем 12GB как total
                                used_vram = sum(m.get("size_vram", 0) for m in running)
                                total_vram = 12 * 1024**3  # RTX 3060 default
                                available_vram_bytes = max(0, total_vram - used_vram)
                                logger.info(f"   Free VRAM (via Ollama): {available_vram_bytes/(1024**3):.1f} GB")
                            else:
                                available_vram_bytes = 8 * 1024**3
                    except Exception:
                        available_vram_bytes = 8 * 1024**3
            except Exception as e:
                logger.warning(f"   VRAM check failed: {e}")
                available_vram_bytes = 8 * 1024**3

            # Шаг 3: умная стратегия загрузки
            coordinator_size = sizes.get(coordinator, 0)
            generator_size = sizes.get(generator, 0)
            utility_size = sizes.get(utility, 0) if utility else 0
            overhead = 1.25  # 25% overhead для kv cache и compute graph

            all_three = (coordinator_size + generator_size + utility_size) * overhead
            both_main = (coordinator_size + generator_size) * overhead

            free_gb = available_vram_bytes / (1024**3)
            logger.info(f"   Need (all 3): {all_three/(1024**3):.1f} GB | Need (2): {both_main/(1024**3):.1f} GB | Free: {free_gb:.1f} GB")

            if utility and all_three <= available_vram_bytes:
                logger.info(f"🟢 All 3 models fit! Loading coordinator + generator + utility")
                load_list = [coordinator, generator, utility]
            elif both_main <= available_vram_bytes:
                logger.info(f"🟡 Loading coordinator + generator (utility will load on-demand)")
                if utility:
                    logger.info(f"   Utility ({utility}) will load on GPU when space frees up")
                load_list = [coordinator, generator]
            else:
                logger.info(f"🟠 Low VRAM: loading generator only")
                load_list = [generator]

            for model in load_list:
                try:
                    resp = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model, "prompt": "", "keep_alive": -1},
                        timeout=120.0
                    )
                    if resp.status_code == 200:
                        logger.info(f"   ✅ {model} loaded (permanent)")
                    else:
                        logger.warning(f"   ⚠️ {model}: {resp.status_code}")
                except Exception as e:
                    logger.warning(f"   ⚠️ {model}: {e}")

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
