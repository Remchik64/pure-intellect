"""System endpoints: health, config, hardware, CCI, logs, and OpenAI-compatible API."""

import logging
import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..api.schemas import HealthResponse
from ..api.state import get_pipeline, get_model_manager, LOG_BUFFER, LOG_LOCK
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────────

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


# ── Config & Hardware ──────────────────────────────────────────────────────────

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


# ── CCI ─────────────────────────────────────────────────────────────────────────

@router.get("/cci/stats")
async def cci_stats():
    """Статистика Context Coherence Index."""
    try:
        pipeline = get_pipeline()
        return pipeline.cci_tracker.stats()
    except Exception as e:
        logger.error(f"CCI stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Logs ────────────────────────────────────────────────────────────────────────

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
    with LOG_LOCK:
        all_lines = list(LOG_BUFFER)
    
    # Фильтрация по уровню
    level_upper = level.upper()
    level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    if level_upper != "ALL" and level_upper in level_order:
        min_level = level_order[level_upper]
        all_lines = [e for e in all_lines if level_order.get(e.get("level", "DEBUG"), 0) >= min_level]
    
    total = len(all_lines)
    # берём последние limit строк
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
    with LOG_LOCK:
        LOG_BUFFER.clear()
    return {"status": "cleared"}


# ── OpenAI-Compatible API ─────────────────────────────────────────────────────

import json as _json_module
import time
import uuid


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "contextor"
    messages: list[OpenAIMessage]
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False


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
    try:
        # ── РЕЖИМ 1: Ollama proxy (любая не-CTX модель) ────────────────────────
        pi_models = {"contextor", "contextor-code", "contextor-fast"}
        if req.model not in pi_models:
            ollama_payload = {
                "model": req.model,
                "messages": [m.dict() for m in req.messages],
                "temperature": req.temperature,
                "stream": False,
                "options": {"num_ctx": 4096, "num_gpu": -1},
            }
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{settings.ollama_url}/v1/chat/completions",
                    json=ollama_payload,
                )
                if resp.status_code == 200:
                    return resp.json()
                raise HTTPException(status_code=resp.status_code, detail=f"Ollama proxy failed: {resp.text[:200]}")

        # ── РЕЖИМ 2: Contextor pipeline с памятью ────────────────────────
        pipe = get_pipeline()
        if pipe is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        query = user_messages[-1].content
        system_messages = [m for m in req.messages if m.role == "system"]
        system_override = system_messages[0].content if system_messages else None

        result = pipe.run(
            query=query,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            system=system_override,
        )
        response_text = result.response
        prompt_tokens = result.tokens_prompt or len(query.split()) * 2
        completion_tokens = result.tokens_completion or len(response_text.split()) * 2

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
