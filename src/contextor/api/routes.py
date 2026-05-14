"""API endpoints."""

import collections
import datetime
import threading
import httpx
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..api.schemas import ChatRequest, ChatResponse, HealthResponse
from ..engine import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


# Singleton pipeline — сохраняет память между запросами
_pipeline = None
_pipeline_lock = threading.Lock()

# ── Download progress tracking ────────────────────────────────────────────────
# model_name → {"status": str, "percent": int, "speed": str, "error": str|None}
_download_progress: dict[str, dict] = {}

# ── In-memory log buffer ──────────────────────────────────────────────────────
_LOG_BUFFER: collections.deque = collections.deque(maxlen=2000)
_LOG_LOCK = threading.Lock()

class _PIMemoryHandler(logging.Handler):
    """Перехватывает все logging записи в _LOG_BUFFER."""
    def emit(self, record: logging.LogRecord):
        try:
            ts = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
            line = f"[{ts}] {record.levelname:<8} {record.name}: {record.getMessage()}"
            if record.exc_info:
                import traceback as _tb
                line += "\n" + "".join(_tb.format_exception(*record.exc_info))
            with _LOG_LOCK:
                _LOG_BUFFER.append({"ts": ts, "level": record.levelname, "name": record.name, "line": line})
        except Exception:
            pass

# Прикрепляем к root logger чтобы перехватывать ВСЕ логи
_mem_handler = _PIMemoryHandler()
_mem_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_mem_handler)


def get_model_manager() -> ModelManager:
    """Получить thread-safe singleton ModelManager."""
    return ModelManager.get_instance(cache_dir="./models")


def get_pipeline():
    """Получить thread-safe singleton OrchestratorPipeline."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from ..core import OrchestratorPipeline
                _pipeline = OrchestratorPipeline(model_manager=get_model_manager())
    return _pipeline


@router.get("/health", response_model=HealthResponse)
async def health():
    """Проверка здоровья сервера."""
    manager = get_model_manager()
    loaded = manager.loaded_model is not None
    return HealthResponse(
        status="healthy",
        model_loaded=loaded,
        version="0.1.0"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Отправить сообщение модели."""
    manager = get_model_manager()
    
    # Загрузить модель если не загружена
    if manager.loaded_model is None:
        try:
            model_key = request.model or "qwen3.5-2b"
            manager.load(model_key, n_gpu_layers=-1)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    try:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.query})
        
        response = manager.chat(
            messages=messages,
            temperature=request.temperature or 0.7,
        )
        
        return ChatResponse(
            response=response,
            model=request.model or "qwen3.5-2b",
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
@router.get("/memory/facts")
async def memory_facts():
    try:
        pipeline = get_pipeline()
        wm_facts = [f.__dict__ for f in pipeline.working_memory._facts]
        storage_facts = [f.__dict__ for f in pipeline.memory_storage._facts]
        return {"facts": wm_facts + storage_facts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def memory_stats():
    """Статистика самообновляемой памяти."""
    try:
        pipeline = get_pipeline()
        return pipeline.memory_stats()
    except Exception as e:
        logger.error(f"Memory stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/clear")
async def memory_clear():
    """Очистить рабочую память (переместить все факты в долгосрочное хранилище)."""
    try:
        pipeline = get_pipeline()
        pipeline.memory_clear()
        return {"status": "cleared", "message": "Working memory cleared, facts moved to storage"}
    except Exception as e:
        logger.error(f"Memory clear failed: {e}")


@router.get("/cci/stats")
async def cci_stats():
    """Статистика Context Coherence Index."""
    try:
        pipeline = get_pipeline()
        return pipeline.cci_tracker.stats()
    except Exception as e:
        logger.error(f"CCI stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/info")
async def session_info():
    """Информация о текущей сохранённой сессии."""
    try:
        pipeline = get_pipeline()
        return pipeline.session_info()
    except Exception as e:
        logger.error(f"Session info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session")
async def session_delete():
    """Удалить сохранённую сессию и сбросить состояние."""
    try:
        pipeline = get_pipeline()
        pipeline.session_delete()
        return {"status": "deleted", "message": "Session deleted and state reset"}
    except Exception as e:
        logger.error(f"Session delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dual-model/stats")
async def dual_model_stats():
    """Статистика Dual Model Router (P6): coordinator vs generator."""
    try:
        pipeline = get_pipeline()
        return pipeline.dual_model_stats()
    except Exception as e:
        logger.error(f"Dual model stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Config & Hardware endpoints (Шаг 1: гибкая система моделей) ─────────────

@router.get("/config")
async def config_info():
    """Текущая конфигурация моделей из config.yaml."""
    try:
        pipeline = get_pipeline()
        return pipeline._router.config_info()
    except Exception as e:
        logger.error(f"Config info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/reload")
async def config_reload():
    """Перечитать config.yaml и обновить модели без перезапуска.

    Позволяет сменить модель в config.yaml и применить изменения
    без остановки сервера.
    """
    try:
        pipeline = get_pipeline()
        old_coordinator = pipeline._router.coordinator_model
        old_generator = pipeline._router.generator_model
        pipeline._router.reload_from_config()
        # Сброс кеша ProviderFactory
        from contextor.engines.provider import ProviderFactory
        ProviderFactory.reset()
        return {
            "status": "reloaded",
            "coordinator": {
                "before": old_coordinator,
                "after": pipeline._router.coordinator_model,
                "changed": old_coordinator != pipeline._router.coordinator_model,
            },
            "generator": {
                "before": old_generator,
                "after": pipeline._router.generator_model,
                "changed": old_generator != pipeline._router.generator_model,
            },
        }
    except Exception as e:
        logger.error(f"Config reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/hardware")
async def hardware_info():
    """Информация об аппаратных ресурсах: VRAM, GPU, оптимальные слои."""
    try:
        from contextor.engines.provider import (
            detect_free_vram_mb,
            detect_optimal_gpu_layers,
        )
        from contextor.engines.config_loader import get_config
        cfg = get_config()
        free_vram = detect_free_vram_mb()
        return {
            "gpu_available": free_vram > 0,
            "free_vram_mb": free_vram,
            "free_vram_gb": round(free_vram / 1024, 2),
            "optimal_gpu_layers": {
                "3b_model": detect_optimal_gpu_layers(model_size_gb=2.0),
                "7b_model": detect_optimal_gpu_layers(model_size_gb=4.7),
                "13b_model": detect_optimal_gpu_layers(model_size_gb=8.0),
            },
            "config": {
                "auto_gpu_layers": cfg.hardware.auto_gpu_layers,
                "vram_reserve_mb": cfg.hardware.vram_reserve_mb,
                "vram_overflow_strategy": cfg.hardware.vram_overflow_strategy,
                "cpu_threads": cfg.hardware.cpu_threads,
            },
            "hint": (
                "GPU available — используй gpu_layers: -1 для максимальной скорости"
                if free_vram > 4096
                else "Мало VRAM — используй gpu_layers: auto для частичного offload на CPU"
                if free_vram > 0
                else "GPU не обнаружен — используй gpu_layers: 0 (CPU only)"
            ),
        }
    except Exception as e:
        logger.error(f"Hardware info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Admin Panel: дополнительные endpoints ──────────────────

@router.get("/ollama/models")
async def ollama_models_proxy():
    """Прокси к Ollama API — список доступных моделей.

    Решает CORS проблему браузера при обращении к localhost:11434.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://host.docker.internal:11434/api/tags")
            return resp.json()
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        return {"models": [], "error": str(e)}


@router.delete("/models/{model_name:path}")
async def delete_model(model_name: str):
    """Удалить модель из Ollama полностью (освобождает место на диске)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method="DELETE",
                url="http://host.docker.internal:11434/api/delete",
                json={"name": model_name},
            )
            if resp.status_code == 200:
                logger.info(f"[admin] Model deleted: {model_name}")
                return {"status": "deleted", "model": model_name}
            else:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Ollama error: {resp.text[:200]}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def models_status():
    """Полный статус всех моделей: скачанные + активные в VRAM.

    Возвращает:
    - downloaded: все скачанные модели (из /api/tags)
    - active_in_vram: модели загруженные в VRAM прямо сейчас (из /api/ps)
    - coordinator/generator: статус назначенных моделей

    Статусы:
    - 'active'  — загружена в VRAM прямо сейчас 🔥
    - 'ready'   — скачана, готова к запуску ✅
    - 'offline' — не скачана, нужно скачать ❌
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Все скачанные модели
            tags_resp = await client.get("http://host.docker.internal:11434/api/tags")
            tags_resp.raise_for_status()
            downloaded = [m["name"] for m in tags_resp.json().get("models", [])]

            # Активные в VRAM прямо сейчас
            try:
                ps_resp = await client.get("http://host.docker.internal:11434/api/ps")
                active = [m["name"] for m in ps_resp.json().get("models", [])]
            except Exception:
                active = []

            # Назначенные модели из пайплайна
            try:
                pipeline = get_pipeline()
                coordinator = pipeline._router.coordinator_model
                generator = pipeline._router.generator_model
            except Exception:
                coordinator = "qwen3.5:2b"
                generator = "qwen3.5:9b"

            def get_status(model_name: str) -> str:
                if model_name in active:
                    return "active"   # загружена в VRAM
                elif model_name in downloaded:
                    return "ready"    # скачана, готова
                else:
                    return "offline"  # не скачана

            return {
                "downloaded": downloaded,
                "active_in_vram": active,
                "coordinator": {
                    "model": coordinator,
                    "status": get_status(coordinator),
                },
                "generator": {
                    "model": generator,
                    "status": get_status(generator),
                },
            }
    except Exception as e:
        logger.warning(f"Models status failed: {e}")
        return {"error": str(e), "downloaded": [], "active_in_vram": [],
                "coordinator": {"model": "—", "status": "offline"},
                "generator": {"model": "—", "status": "offline"}}


@router.post("/models/switch")
async def switch_model(req: dict):
    """Переключить модель координатора или генератора без перезапуска.

    Body: {"role": "coordinator" | "generator", "model": "qwen3.5:9b"}
    """
    try:
        pipeline = get_pipeline()
        role = req.get("role")
        model = req.get("model")
        if role not in ("coordinator", "generator"):
            raise HTTPException(status_code=400, detail="role must be coordinator or generator")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")

        router_obj = pipeline._router
        if role == "coordinator":
            router_obj.coordinator_model = model
            logger.info(f"[admin] Coordinator switched to {model}")
        else:
            router_obj.generator_model = model
            logger.info(f"[admin] Generator switched to {model}")

        # Сохраняем в config.yaml — чтобы выбор сохранялся после перезапуска
        try:
            from contextor.engines.config_loader import save_model_to_config
            saved = save_model_to_config(role, model)
            save_status = "saved to config.yaml" if saved else "runtime only (config save failed)"
            logger.info(f"[admin] Config {'saved' if saved else 'not saved'}: {role}={model}")
        except Exception as e:
            save_status = f"runtime only ({e})"
            logger.warning(f"[admin] Could not save config: {e}")

        return {"status": "switched", "role": role, "model": model, "persistence": save_status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Switch model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── OpenAI-Compatible API ─────────────────────────────────────────────────
# Позволяет использовать Contextor как OpenAI-совместимый сервер


import json as _json_module


async def _sse_stream(content: str, model: str, req_id: str):
    """Fake SSE streaming для OpenAI-совместимых клиентов."""
    words = content.split(" ")
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
        }
        yield f"data: {_json_module.dumps(chunk, ensure_ascii=False)}\n\n"
    final = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {_json_module.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


openai_router = APIRouter(prefix="/v1", tags=["openai-compatible"])


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "contextor"
    messages: list[OpenAIMessage]
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False


@openai_router.get("/models")
async def openai_list_models():
    """Список доступных моделей (OpenAI формат)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "contextor",
                "object": "model",
                "created": 1714000000,
                "owned_by": "contextor",
                "description": "Local AI with hierarchical memory",
            },
            {
                "id": "contextor-code",
                "object": "model",
                "created": 1714000000,
                "owned_by": "contextor",
                "description": "Contextor with Code Module",
            },
        ],
    }


@openai_router.post("/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest):
    """OpenAI-совместимый endpoint для чата.

    Два режима работы:
    1. model='contextor' → полный pipeline с иерархической памятью
    2. Любая другая модель (напр. 'qwen3.5:2b') → Ollama proxy без памяти
    """
    import time
    import uuid

    try:
        # ── РЕЖИМ 1: Ollama proxy (любая не-CTX модель) ────────────────────────
        # Если запрашивают не contextor модель — проксируем к Ollama напрямую
        pi_models = {"contextor", "contextor-code", "contextor-fast"}
        if req.model not in pi_models:
            # Прямой proxy к Ollama — модель передаётся как есть
            ollama_payload = {
                "model": req.model,
                "messages": [m.dict() for m in req.messages],
                "temperature": req.temperature,
                "stream": False,
                "options": {"num_ctx": 4096, "num_gpu": -1},
            }
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    "http://host.docker.internal:11434/v1/chat/completions",
                    json=ollama_payload,
                )
                if resp.status_code == 200:
                    return resp.json()
                raise HTTPException(status_code=resp.status_code, detail=f"Ollama proxy failed: {resp.text[:200]}")


        # ── РЕЖИМ 2: Contextor pipeline с памятью ────────────────────────
        pipe = get_pipeline()
        if pipe is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Извлекаем последнее user сообщение как основной запрос
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        query = user_messages[-1].content

        # Если есть system message — используем как кастомный промпт
        system_messages = [m for m in req.messages if m.role == "system"]
        system_override = system_messages[0].content if system_messages else None

        # ── Contextor pipeline с памятью ─────────────────────────────────────
        result = pipe.run(
            query=query,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            system=system_override,
        )
        response_text = result.response
        prompt_tokens = result.tokens_prompt or len(query.split()) * 2
        completion_tokens = result.tokens_completion or len(response_text.split()) * 2

        # Возвращаем в формате OpenAI
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        response_body = {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": "contextor-v1",
        }
        # Streaming: возвращаем SSE если клиент запросил stream=True
        if req.stream:
            return StreamingResponse(
                _sse_stream(response_text, req.model, req_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return response_body

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Hardware Detection ────────────────────────────────────────────────────────

@router.get("/hardware/detect")
async def hardware_detect():
    """Определяет железо пользователя и возвращает рекомендации по моделям."""
    try:
        from contextor.utils.hardware_detector import detect_hardware
        return detect_hardware()
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        return {
            "hardware": {"os": "unknown", "ram_gb": 0, "gpu": None},
            "recommendation": {
                "coordinator": "qwen3.5:2b",
                "generator": "qwen3.5:2b",
                "mode": "CPU ONLY",
                "speed_estimate": "~2 tok/sec",
                "status": "⚠️",
                "status_label": "Не определено",
                "num_gpu": 0,
                "warnings": [str(e)],
                "notes": "",
            },
            "errors": [str(e)],
        }


# ── Model Download (через Ollama) ─────────────────────────────────────────────

@router.post("/models/download")
async def download_model(req: dict):
    """Скачать модель через Ollama со streaming прогрессом.

    Body: {"model": "qwen3.5:2b"}
    Прогресс доступен через GET /models/download/check/{model}
    """
    import asyncio
    import json as _json
    import time as _time

    model = req.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    # Если уже качается — не запускаем повторно
    if _download_progress.get(model, {}).get("status") == "downloading":
        return {"status": "already_downloading", "model": model}

    _download_progress[model] = {
        "status": "starting",
        "percent": 0,
        "speed": "",
        "error": None,
        "started_at": _time.time(),
    }

    async def _pull_with_progress():
        """Streaming pull через Ollama HTTP API с парсингом прогресса."""
        _download_progress[model]["status"] = "downloading"
        last_speed_check = _time.time()
        last_completed = 0

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "http://host.docker.internal:11434/api/pull",
                    json={"name": model, "stream": True},
                    timeout=None,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue

                        status_msg = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)

                        # Вычисляем процент
                        percent = 0
                        if total and total > 0:
                            percent = int(completed / total * 100)

                        # Вычисляем скорость (bytes/sec)
                        now = _time.time()
                        elapsed = now - last_speed_check
                        speed_str = ""
                        if elapsed >= 1.0 and completed > last_completed:
                            bytes_per_sec = (completed - last_completed) / elapsed
                            if bytes_per_sec >= 1_048_576:
                                speed_str = f"{bytes_per_sec / 1_048_576:.1f} MB/s"
                            elif bytes_per_sec >= 1024:
                                speed_str = f"{bytes_per_sec / 1024:.1f} KB/s"
                            else:
                                speed_str = f"{bytes_per_sec:.0f} B/s"
                            last_speed_check = now
                            last_completed = completed

                        _download_progress[model].update({
                            "status": "downloading",
                            "percent": percent,
                            "speed": speed_str,
                            "status_msg": status_msg,
                            "completed": completed,
                            "total": total,
                            "error": None,
                        })

                        # Ollama сигнализирует об успехе через status == "success"
                        if status_msg == "success":
                            break

            _download_progress[model]["status"] = "done"
            _download_progress[model]["percent"] = 100
            _download_progress[model]["speed"] = ""
            logger.info(f"[model_download] {model} downloaded successfully")

        except Exception as e:
            _download_progress[model]["status"] = "error"
            _download_progress[model]["error"] = str(e)
            logger.error(f"[model_download] {model} failed: {e}")

    asyncio.create_task(_pull_with_progress())
    return {"status": "downloading", "model": model, "message": f"Скачивание {model} запущено"}


@router.get("/models/download/check/{model_name:path}")
async def check_model_downloaded(model_name: str):
    """Прогресс скачивания и статус модели в Ollama.

    Возвращает реальный прогресс из _download_progress,
    а также проверяет наличие модели в Ollama tags.
    """
    # Сначала отдаём прогресс если есть активное скачивание
    progress = _download_progress.get(model_name)

    # Проверяем готовность через Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://host.docker.internal:11434/api/tags")
            data = resp.json()
            available = [m["name"] for m in data.get("models", [])]
            is_ready = any(
                model_name == m or m.startswith(model_name)
                for m in available
            )
    except Exception as e:
        available = []
        is_ready = False

    if progress is not None:
        return {
            "model": model_name,
            "ready": is_ready or progress.get("status") == "done",
            "status": progress.get("status", "unknown"),
            "percent": progress.get("percent", 0),
            "speed": progress.get("speed", ""),
            "status_msg": progress.get("status_msg", ""),
            "error": progress.get("error"),
            "available_models": available,
        }

    # Нет записи в прогрессе — просто проверяем наличие
    return {
        "model": model_name,
        "ready": is_ready,
        "status": "ready" if is_ready else "not_downloaded",
        "percent": 100 if is_ready else 0,
        "speed": "",
        "status_msg": "",
        "error": None,
        "available_models": available,
    }


# ── Logs endpoint ─────────────────────────────────────────────────────────────

@router.get("/logs")
async def get_logs(
    limit: int = Query(default=500, ge=1, le=2000),
    level: str = Query(default="ALL"),
    offset: int = Query(default=0, ge=0),
):
    """Получить последние N строк логов из memory buffer.
    
    Args:
        limit: максимальное кол-во строк (1-2000)
        level: фильтр уровня ALL / DEBUG / INFO / WARNING / ERROR / CRITICAL
        offset: пропустить первые N строк (для пагинации)
    """
    with _LOG_LOCK:
        all_lines = list(_LOG_BUFFER)
    
    # Фильтрация по уровню
    level_upper = level.upper()
    level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    if level_upper != "ALL" and level_upper in level_order:
        min_level = level_order[level_upper]
        all_lines = [e for e in all_lines if level_order.get(e.get("level", "DEBUG"), 0) >= min_level]
    
    total = len(all_lines)
    # Берём последние limit строк
    lines = all_lines[max(0, total - limit - offset): total - offset if offset else None]
    
    return {
        "total": total,
        "count": len(lines),
        "level_filter": level_upper,
        "lines": lines,
    }


@router.delete("/logs")
async def clear_logs():
    """Очистить буфер логов."""
    with _LOG_LOCK:
        _LOG_BUFFER.clear()
    return {"status": "cleared"}
