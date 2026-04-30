"""API endpoints."""

import os
import collections
import datetime
import threading
import httpx
import logging
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..api.schemas import ChatRequest, ChatResponse, HealthResponse, ModelListResponse, OrchestrateRequest
from ..engine import ModelManager, MODEL_REGISTRY

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




import re as _re


def _strip_thinking(text: str) -> str:
    """Убрать  блоки qwen3/deepseek thinking mode из текста."""
    return _re.sub(r'', '', text, flags=_re.DOTALL | _re.IGNORECASE).strip()


def _make_fallback_json(text: str) -> str:
    """Обернуть текст в валидный JSON response если модель вернула не-JSON."""
    import json as _json
    # Экранируем текст для JSON
    safe_text = _json.dumps(text)[1:-1]  # убираем внешние кавычки
    return '{"thoughts":["Model returned non-JSON response, wrapping as text"],"tool_name":"response","tool_args":{"text":"' + safe_text + '"}}'


def _extract_first_json(text: str) -> str:
    """Извлечь первый полный JSON объект из текста.
    
    Некоторые модели (ministral-3:14b, qwen3.5 и др.) иногда генерируют
     блоки, два JSON объекта подряд, markdown или добавляют
    текст после JSON. Эта функция:
    1. Убирает thinking блоки
    2. Извлекает первый валидный JSON
    3. Если JSON не найден — оборачивает текст в fallback response JSON
    """
    # Убираем thinking блоки qwen3.5 / deepseek перед поиском JSON
    text = _strip_thinking(text)
    start = text.find('{')
    if start == -1:
        # Нет JSON вообще — оборачиваем весь текст как response
        return _make_fallback_json(text)
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    import json as _json
                    _json.loads(candidate)
                    return candidate
                except Exception:
                    # Нашли баланс скобок но JSON невалиден — ищем дальше
                    pass
    # JSON не найден или невалиден — оборачиваем как response
    return _make_fallback_json(text)


def get_model_manager() -> ModelManager:
    """Получить thread-safe singleton ModelManager."""
    return ModelManager.get_instance(cache_dir="./models")
# ── PI Memory Notifications & Coordinator Swap ───────────────────────────────

def _inject_pi_notifications(response_text: str, notifications: list) -> str:
    """Вставить PI уведомления в thoughts поле JSON ответа агента."""
    if not notifications:
        return response_text
    try:
        import json as _j
        data = _j.loads(response_text)
        existing = data.get("thoughts", [])
        if not isinstance(existing, list):
            existing = [str(existing)]
        data["thoughts"] = notifications + existing
        return _j.dumps(data, ensure_ascii=False)
    except Exception:
        return response_text


async def _create_az_coordinate(coordinator_model: str, messages: list) -> str:
    """Вызвать coordinator для создания координаты (snapshot) разговора."""
    recent = messages[-20:] if len(messages) > 20 else messages
    conversation = "\n".join(
        f"{m.get('role','?')}: {str(m.get('content',''))[:300]}"
        for m in recent if m.get('content')
    )
    prompt = (
        "Создай краткую координату (snapshot) разговора. "
        "Максимум 5 предложений. Только факты: тема, участники, ключевые решения, статус.\n\n"
        f"Разговор:\n{conversation}\n\nКоордината:"
    )
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": coordinator_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": 4096, "num_gpu": -1, "keep_alive": -1},
                },
            )
            if resp.status_code == 200:
                coordinate = resp.json().get("response", "").strip()
                return _strip_thinking(coordinate)
    except Exception as e:
        logger.warning(f"[PI Coordinate] Failed: {e}")
    return ""


# Порог сообщений для создания координаты в AZ режиме (~4-5 turns AZ)
_AZ_COORDINATE_MSG_THRESHOLD = 16




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


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """Список доступных моделей."""
    manager = get_model_manager()
    downloaded = manager.list_downloaded()
    models = {}
    for key, info in MODEL_REGISTRY.items():
        models[key] = {
            "name": info["name"],
            "size_gb": info["size_gb"],
            "vram_gb": info["vram_gb"],
            "downloaded": key in downloaded,
            "good_for": info["good_for"],
        }
    return ModelListResponse(models=models)


@router.post("/model/load")
async def load_model(model_key: str = "qwen2.5-3b", gpu_layers: int = -1):
    """Загрузить модель."""
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    
    manager = get_model_manager()
    try:
        llm = manager.load(model_key, n_gpu_layers=gpu_layers)
        info = MODEL_REGISTRY[model_key]
        return {
            "status": "loaded",
            "model": model_key,
            "name": info["name"],
        }
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Отправить сообщение модели."""
    manager = get_model_manager()
    
    # Загрузить модель если не загружена
    if manager.loaded_model is None:
        try:
            model_key = request.model or "qwen2.5-3b"
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
            model=request.model or "qwen2.5-3b",
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intent")
async def detect_intent(request: ChatRequest):
    """Определить намерение запроса."""
    from ..core import IntentDetector
    
    manager = get_model_manager()
    detector = IntentDetector(model_manager=manager)
    
    # Пытаемся загрузить модель если не загружена
    use_llm = False
    if manager.loaded_model is None:
        try:
            manager.load("qwen2.5-3b", n_gpu_layers=-1)
            use_llm = True
        except Exception:
            pass
    else:
        use_llm = True
    
    result = detector.detect(request.query, use_llm=use_llm)
    
    return {
        "intent": result.intent.value,
        "confidence": result.confidence,
        "entities": result.entities,
        "keywords": result.keywords,
        "reasoning": result.reasoning,
        "suggested_context": result.suggested_context,
    }


@router.post("/index")
async def index_project(directory: str = ".", extensions: List[str] = [".py"]):
    """Index project directory for code cards."""
    from ..core import CardGenerator
    
    try:
        generator = CardGenerator()
        total_cards = generator.index_directory(Path(directory), extensions)
        
        return {
            "status": "indexed",
            "directory": directory,
            "total_cards": total_cards,
            "extensions": extensions,
        }
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cards/search")
async def search_cards(query: str, top_k: int = 5):
    """Search code cards by query."""
    from ..core import CardGenerator
    
    try:
        generator = CardGenerator()
        cards = generator.search_cards(query, top_k)
        
        return {
            "query": query,
            "results": [
                {
                    "card_id": card.card_id,
                    "entity_name": card.entity.name,
                    "entity_type": card.entity.type,
                    "file_path": card.entity.file_path,
                    "summary": card.summary,
                }
                for card in cards
            ],
        }
    except Exception as e:
        logger.error(f"Card search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieve")
async def retrieve_context(
    query: str,
    top_k: int = 5,
    intent: Optional[str] = None,
):
    """Retrieve relevant code context using RAG."""
    from ..core import Retriever
    
    try:
        retriever = Retriever()
        
        if intent:
            # Поиск по intent
            results = retriever.search_by_intent(intent)
        else:
            # Обычный поиск
            results = retriever.search(query, top_k=top_k)
        
        return {
            "query": query,
            "intent": intent,
            "total_cards": retriever.count(),
            "results": [
                {
                    "card_id": r.card_id,
                    "entity_name": r.entity_name,
                    "entity_type": r.entity_type,
                    "file_path": r.file_path,
                    "lines": f"{r.start_line}-{r.end_line}",
                    "summary": r.summary,
                    "distance": round(r.distance, 4),
                    "relevance": round(r.relevance_score, 4),
                }
                for r in results
            ],
            "context": retriever.format_context(results),
        }
    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assemble")
async def assemble_context(request: ChatRequest):
    """Assemble context for LLM from user query."""
    from ..core import IntentDetector, ContextAssembler
    
    try:
        # 1. Intent detection
        detector = IntentDetector()
        intent_result = detector.detect(request.query)
        
        # 2. Context assembly
        assembler = ContextAssembler()
        result = assembler.assemble_and_respond(request.query, intent_result)
        
        return {
            "query": request.query,
            "intent": intent_result.intent.value,
            "confidence": intent_result.confidence,
            "entities": intent_result.entities,
            "total_tokens": result["total_tokens"],
            "mode": result["mode"],
            "messages": result["messages"],
        }
    except Exception as e:
        logger.error(f"Assemble failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/build")
async def build_graph(directory: str = "./src", extensions: List[str] = [".py"]):
    """Build knowledge graph from project directory."""
    from ..core import GraphBuilder
    
    try:
        builder = GraphBuilder()
        stats = builder.build_from_directory(Path(directory), extensions)
        return {
            "status": "built",
            "directory": directory,
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Graph build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats")
async def graph_stats():
    """Get knowledge graph statistics."""
    from ..core import GraphBuilder
    
    try:
        builder = GraphBuilder()
        return builder.get_stats()
    except Exception as e:
        logger.error(f"Graph stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/search")
async def graph_search(query: str, limit: int = 10):
    """Search knowledge graph nodes."""
    from ..core import GraphBuilder
    
    try:
        builder = GraphBuilder()
        results = builder.search(query, limit)
        return {
            "query": query,
            "results": results,
        }
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/file")
async def graph_file(file_path: str):
    """Get graph subgraph for a specific file."""
    from ..core import GraphBuilder
    
    try:
        builder = GraphBuilder()
        return builder.get_file_graph(file_path)
    except Exception as e:
        logger.error(f"Graph file failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watcher/start")
async def start_watcher(directory: str = "./src"):
    """Start file watcher for project directory."""
    from ..core import WatcherIntegration
    
    try:
        integration = WatcherIntegration(directory)
        integration.start()
        return {
            "status": "started",
            "directory": directory,
        }
    except Exception as e:
        logger.error(f"Watcher start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watcher/stop")
async def stop_watcher():
    """Stop file watcher."""
    from ..core import WatcherIntegration
    
    try:
        integration = WatcherIntegration()
        integration.stop()
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Watcher stop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watcher/scan")
async def scan_changes(directory: str = "./src"):
    """One-time scan for file changes."""
    from ..core import WatcherIntegration
    
    try:
        integration = WatcherIntegration(directory)
        changes = integration.scan_now()
        return {
            "status": "scanned",
            "directory": directory,
            "changes": changes,
            "total": len(changes),
        }
    except Exception as e:
        logger.error(f"Watcher scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watcher/status")
async def watcher_status():
    """Get watcher status."""
    from ..core import WatcherIntegration
    
    try:
        integration = WatcherIntegration()
        return integration.get_status()
    except Exception as e:
        logger.error(f"Watcher status failed: {e}")


@router.post("/orchestrate")
async def orchestrate(request: OrchestrateRequest):
    """Полный пайплайн: Intent → RAG → Graph → Assembler → LLM."""
    try:
        pipeline = get_pipeline()
        result = pipeline.run(
            query=request.query,
            model_key=request.model,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_llm_intent=request.use_llm_intent,
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
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

class FactSaveRequest(BaseModel):
    text: str
    importance: float = 0.7
    is_anchor: bool = False
    session_id: str = "default"
    metadata: dict = {}


@router.post("/memory/fact")
async def save_memory_fact(request: FactSaveRequest):
    """Сохранить факт в рабочую память PI (для Agent Zero memory bridge)."""
    try:
        pipeline = get_pipeline()
        if request.is_anchor:
            fact = pipeline.working_memory.add_anchor(
                request.text, source="agent_zero"
            )
        else:
            # Фильтруем 'source' из metadata чтобы избежать дублирования
            clean_metadata = {k: v for k, v in request.metadata.items() if k != 'source'}
            fact = pipeline.working_memory.add_text(
                request.text, source="agent_zero",
                importance=request.importance,
                **clean_metadata
            )
        return {"id": fact.fact_id, "status": "saved", "is_anchor": fact.is_anchor}
    except Exception as e:
        logger.error(f"Save memory fact failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/search")
async def search_memory_facts(
    query: str,
    limit: int = 10,
    session_id: str = "default"
):
    """Поиск фактов в памяти PI (для Agent Zero memory bridge)."""
    try:
        pipeline = get_pipeline()
        # Поиск в долгосрочном хранилище
        storage_results = pipeline.memory_storage.retrieve(query, top_k=limit)
        # Также берём из рабочей памяти (горячие факты)
        wm_facts = pipeline.working_memory.get_facts()
        # Объединяем - рабочая память приоритетнее
        seen_ids = set()
        results = []
        for fact in wm_facts:
            if fact.fact_id not in seen_ids and query.lower() in fact.content.lower():
                results.append({
                    "id": fact.fact_id,
                    "text": fact.content,
                    "score": fact.attention_weight,
                    "is_anchor": fact.is_anchor,
                    "metadata": {"source": "working_memory"}
                })
                seen_ids.add(fact.fact_id)
        for fact in storage_results:
            if fact.fact_id not in seen_ids:
                results.append({
                    "id": fact.fact_id,
                    "text": fact.content,
                    "score": fact.attention_weight,
                    "is_anchor": fact.is_anchor,
                    "metadata": {"source": "storage"}
                })
                seen_ids.add(fact.fact_id)
        return {"results": results[:limit], "total": len(results)}
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/cci/stats")
async def cci_stats():
    """Статистика Context Coherence Index."""
    try:
        pipeline = get_pipeline()
        return pipeline.cci_tracker.stats()
    except Exception as e:
        logger.error(f"CCI stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cci/reset")
async def cci_reset():
    """Сбросить историю CCI (новая сессия)."""
    try:
        pipeline = get_pipeline()
        pipeline.cci_tracker.reset()
        return {"status": "reset", "message": "CCI history cleared"}
    except Exception as e:
        logger.error(f"CCI reset failed: {e}")


@router.get("/coordinates")
async def get_coordinates():
    """Информация о текущих координатах сессии (для UI)."""
    try:
        pipeline = get_pipeline()
        return pipeline.coordinates_info()
    except Exception as e:
        logger.error(f"Coordinates info failed: {e}")
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


@router.post("/session/save")
async def session_save():
    """Принудительно сохранить текущую сессию."""
    try:
        pipeline = get_pipeline()
        pipeline._session.save(
            working_memory=pipeline.working_memory,
            storage=pipeline.memory_storage,
            chat_history=pipeline._chat_history,
            turn=pipeline._turn,
        )
        return {
            "status": "saved",
            "turn": pipeline._turn,
            "working_memory_facts": pipeline.working_memory.size(),
            "storage_facts": pipeline.memory_storage.size(),
        }
    except Exception as e:
        logger.error(f"Session save failed: {e}")
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


@router.post("/dual-model/refresh")
async def dual_model_refresh():
    """Перепроверить доступность generator модели (7B)."""
    try:
        pipeline = get_pipeline()
        available = pipeline._router.refresh_generator_check()
        return {
            "generator_model": pipeline._router.generator_model,
            "generator_available": available,
            "coordinator_model": pipeline._router.coordinator_model,
        }
    except Exception as e:
        logger.error(f"Dual model refresh failed: {e}")
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
        from pure_intellect.engines.provider import ProviderFactory
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
        from pure_intellect.engines.provider import (
            detect_free_vram_mb,
            detect_optimal_gpu_layers,
        )
        from pure_intellect.engines.config_loader import get_config
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


# ── F1: Multi-session API endpoints ──────────────────────

class NewSessionRequest(BaseModel):
    display_name: Optional[str] = None
    session_type: str = "chat"  # 'chat' or 'project'
    project_path: Optional[str] = None


class RenameSessionRequest(BaseModel):
    display_name: str


@router.get("/sessions")
async def list_sessions():
    """Список всех сессий."""
    try:
        pipeline = get_pipeline()
        return pipeline.get_sessions()
    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/new")
async def create_new_session(req: NewSessionRequest):
    """Создать новую сессию и переключиться на неё."""
    try:
        pipeline = get_pipeline()
        result = pipeline.create_new_session(
            display_name=req.display_name,
            session_type=req.session_type,
            project_path=req.project_path,
        )
        return result
    except Exception as e:
        logger.error(f"Create session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/switch")
async def switch_session(session_id: str):
    """Переключить активную сессию."""
    try:
        pipeline = get_pipeline()
        result = pipeline.switch_session(session_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Session not found"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Switch session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}/rename")
async def rename_session(session_id: str, req: RenameSessionRequest):
    """Переименовать сессию."""
    try:
        pipeline = get_pipeline()
        result = pipeline.rename_session(session_id, req.display_name)
        return result
    except Exception as e:
        logger.error(f"Rename session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Удалить сессию."""
    try:
        pipeline = get_pipeline()
        result = pipeline.delete_session_by_id(session_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail="Cannot delete this session")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── C1: Code Module API endpoints ────────────────────────

class IndexProjectRequest(BaseModel):
    project_path: str
    session_id: Optional[str] = None
    extensions: Optional[list] = None
    force: bool = False


class CodeSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    entity_types: Optional[list] = None


@router.post("/code/index")
async def index_project(req: IndexProjectRequest):
    """Проиндексировать проект в ChromaDB."""
    try:
        from pure_intellect.core.code_module import CodeModule
        module = CodeModule(
            project_path=req.project_path,
            session_id=req.session_id or pipeline._session_manager.active_session_id,
        )
        # Сохраняем в pipeline для последующих запросов
        pipeline._code_module = module
        result = module.index_project(
            extensions=req.extensions,
            force=req.force,
        )
        # Обновляем метаданные сессии
        if result.get("status") == "success":
            pipeline._session_manager.update_meta(
                session_id=pipeline._session_manager.active_session_id,
                indexed_files=result.get("indexed_files", 0),
            )
        return result
    except Exception as e:
        logger.error(f"Index project failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code/search")
async def search_code(q: str, top_k: int = 5):
    """Семантический поиск по проиндексированному коду."""
    try:
        if not hasattr(pipeline, '_code_module') or pipeline._code_module is None:
            raise HTTPException(status_code=404, detail="No project indexed. Use POST /code/index first.")
        results = pipeline._code_module.search(query=q, top_k=top_k)
        return {
            "query": q,
            "results": [r.to_dict() for r in results],
            "total": len(results),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Code search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/code/stats")
async def code_stats():
    """Статистика Code Module."""
    try:
        pipeline = get_pipeline()
        if not hasattr(pipeline, '_code_module') or pipeline._code_module is None:
            return {"active": False, "message": "No project indexed"}
        return {"active": True, **pipeline._code_module.stats()}
    except Exception as e:
        logger.error(f"Code stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/code/graph")
async def build_code_graph():
    """Построить граф зависимостей проекта."""
    try:
        if not hasattr(pipeline, '_code_module') or pipeline._code_module is None:
            raise HTTPException(status_code=404, detail="No project indexed")
        result = pipeline._code_module.build_graph()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Build graph failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Watcher (C2) — авто-индексация при изменениях файлов ──────

@router.get("/code/watcher/status")
async def code_watcher_status():
    """Статус файлового watcher — активен ли мониторинг."""
    pipeline = get_pipeline()
    if not pipeline._code_module:
        return {"is_running": False, "message": "No active project session"}
    return pipeline._code_module.watcher_status()


@router.post("/code/watcher/start")
async def code_watcher_start():
    """Запустить авто-мониторинг изменений файлов проекта.

    При изменении .py файла:
    1. Файл автоматически переиндексируется в ChromaDB
    2. Факт об изменении добавляется в WorkingMemory
    3. Граф зависимостей обновляется
    """
    pipeline = get_pipeline()
    if not pipeline._code_module:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No active project session. Open a project first.")

    def _watcher_callback(file_path: str, event_type: str, summary: str):
        """При изменении файла → факт в WorkingMemory."""
        try:
            pipeline.working_memory.add(
                content=summary,
                source="watcher",
                importance=0.7,
            )
            logger.info(f"[watcher→memory] {summary}")
        except Exception as e:
            logger.error(f"[watcher→memory] failed: {e}")

    result = pipeline._code_module.start_watcher(on_change_callback=_watcher_callback)
    return result


@router.post("/code/watcher/stop")
async def code_watcher_stop():
    """Остановить мониторинг изменений файлов."""
    pipeline = get_pipeline()
    if not pipeline._code_module:
        return {"status": "no_active_project"}
    return pipeline._code_module.stop_watcher()


@router.get("/code/watcher/changes")
async def code_watcher_changes(limit: int = 20):
    """Последние изменения файлов обнаруженные watcher."""
    pipeline = get_pipeline()
    if not pipeline._code_module or not pipeline._code_module._watcher:
        return {"changes": [], "total": 0}
    status = pipeline._code_module.watcher_status()
    changes = status.get("recent_changes", [])[-limit:]
    return {"changes": changes, "total": status.get("total_changes", 0)}


@router.post("/code/watcher/scan")
async def code_watcher_scan():
    """Одноразовое сканирование изменений без запуска непрерывного мониторинга.

    Полезно для проверки что изменилось с момента последней индексации.
    """
    pipeline = get_pipeline()
    if not pipeline._code_module:
        return {"changes": [], "message": "No active project"}
    changes = pipeline._code_module.scan_changes_now()
    return {"changes": changes, "total": len(changes)}


# ── Admin Panel: дополнительные endpoints ──────────────────

@router.get("/ollama/models")
async def ollama_models_proxy():
    """Прокси к Ollama API — список доступных моделей.

    Решает CORS проблему браузера при обращении к localhost:11434.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
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
                url="http://localhost:11434/api/delete",
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
            tags_resp = await client.get("http://localhost:11434/api/tags")
            tags_resp.raise_for_status()
            downloaded = [m["name"] for m in tags_resp.json().get("models", [])]

            # Активные в VRAM прямо сейчас
            try:
                ps_resp = await client.get("http://localhost:11434/api/ps")
                active = [m["name"] for m in ps_resp.json().get("models", [])]
            except Exception:
                active = []

            # Назначенные модели из пайплайна
            try:
                pipeline = get_pipeline()
                coordinator = pipeline._router.coordinator_model
                generator = pipeline._router.generator_model
            except Exception:
                coordinator = "qwen2.5:3b"
                generator = "qwen2.5:7b"

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

    Body: {"role": "coordinator" | "generator", "model": "qwen2.5:7b"}
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
            from pure_intellect.engines.config_loader import save_model_to_config
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


@router.post("/models/warm")
async def warm_model(req: dict):
    """Прогреть модель в VRAM с keep_alive=-1.

    Body: {"model": "qwen2.5:3b", "role": "utility"}  # role опционально
    Используется Admin Panel для принудительной загрузки utility model в GPU.
    """
    import httpx
    model = req.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Проверяем что модель существует
            show = await client.post(
                "http://localhost:11434/api/show",
                json={"name": model}
            )
            if show.status_code != 200:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not found in Ollama")

            size_gb = show.json().get("size", 0) / (1024**3)

            # Загружаем в VRAM с keep_alive=-1
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": "", "keep_alive": -1},
                timeout=120.0
            )
            if resp.status_code == 200:
                logger.info(f"[warm] ✅ {model} ({size_gb:.1f} GB) loaded to GPU (permanent)")
                return {
                    "status": "loaded",
                    "model": model,
                    "size_gb": round(size_gb, 2),
                    "keep_alive": -1,
                    "message": f"Model '{model}' loaded to GPU permanently"
                }
            else:
                raise HTTPException(status_code=502, detail=f"Ollama returned {resp.status_code}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[warm] Failed to warm {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/fact/{fact_id}")
async def delete_memory_fact(fact_id: str):
    """Удалить конкретный факт из памяти по ID."""
    try:
        pipeline = get_pipeline()
        wm = pipeline.working_memory
        # Удаляем из WorkingMemory
        initial_size = wm.size()
        wm._facts = [f for f in wm._facts if f.id != fact_id]
        deleted_wm = initial_size - wm.size()

        # Удаляем из MemoryStorage
        deleted_storage = 0
        storage = pipeline.memory_storage
        initial_storage = len(storage._facts)
        storage._facts = [f for f in storage._facts if f.id != fact_id]
        deleted_storage = initial_storage - len(storage._facts)

        if deleted_wm + deleted_storage == 0:
            raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")

        return {"deleted": True, "fact_id": fact_id, 
                "from_working_memory": deleted_wm,
                "from_storage": deleted_storage}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete fact failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ── OpenAI-Compatible API (для Agent Zero, Open WebUI, LM Studio) ─────
# Позволяет использовать Pure Intellect как OpenAI-совместимый сервер
# Agent Zero config: api_base = http://localhost:7860/v1


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
    model: str = "pure-intellect"
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
                "id": "pure-intellect",
                "object": "model",
                "created": 1714000000,
                "owned_by": "pure-intellect",
                "description": "Local AI with hierarchical memory",
            },
            {
                "id": "pure-intellect-code",
                "object": "model",
                "created": 1714000000,
                "owned_by": "pure-intellect",
                "description": "Pure Intellect with Code Module",
            },
        ],
    }


@openai_router.post("/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest):
    """OpenAI-совместимый endpoint для чата.

    Два режима работы:
    1. model='pure-intellect' → полный pipeline с иерархической памятью
    2. Любая другая модель (напр. 'qwen2.5:3b') → Ollama proxy без памяти

    Настройка Agent Zero (из Docker контейнера):
      chat_model api_base:    http://host.docker.internal:7860/v1
      chat_model model:       pure-intellect
      utility_model api_base: http://host.docker.internal:7860/v1
      utility_model model:    qwen2.5:3b  (прямой proxy к Ollama)
    """
    import time
    import uuid

    try:
        # ── РЕЖИМ 1: Ollama proxy (utility_model, любая не-PI модель) ──────────
        # Если запрашивают не pure-intellect модель — проксируем к Ollama напрямую
        # Умная эскалация: если запрос большой (документ, tool result) → используем
        # сильную модель вместо слабой утилитарной, иначе мелкие задачи идут на быструю
        pi_models = {"pure-intellect", "pure-intellect-code", "pure-intellect-fast"}
        if req.model not in pi_models:
            # Прямой proxy к Ollama — никакой эскалации
            # utility_model Agent Zero получает ИМЕННО ту модель которую запросила
            ollama_payload = {
                "model": req.model,
                "messages": [m.dict() for m in req.messages],
                "temperature": req.temperature,
                "stream": False,
                "options": {"num_ctx": 8192, "num_gpu": -1},
            }
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    "http://localhost:11434/v1/chat/completions",
                    json=ollama_payload,
                )
                if resp.status_code == 200:
                    return resp.json()
                raise HTTPException(status_code=resp.status_code, detail=f"Ollama proxy failed: {resp.text[:200]}")


        # ── РЕЖИМ 2: Pure Intellect pipeline с памятью ────────────────────────
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

        # ── AGENT ZERO режим: детектируем по system_override ───────────────────
        # Agent Zero ВСЕГДА передаёт system_override (свой системный промпт).
        # Мы должны передавать ВСЕ messages[] напрямую в Ollama — иначе:
        #   - pipe.run() добавит PI _chat_history → дублирование
        #   - ответ будет текстом, не JSON → Agent Zero misformat → цикл
        # Признак Agent Zero: длинный system_override (>500 символов)
        is_agent_zero = (
            system_override is not None
            and len(system_override) > 500  # Agent Zero system prompt ~6000 символов
        )

        if is_agent_zero:
            # ПРОЗРАЧНЫЙ ПРОКСИ для Agent Zero
            # Единственное добавление — компактный контекст памяти PI к system prompt
            # Agent Zero сам парсит JSON ответ и запускает инструменты (EXE)
            try:
                all_facts = pipe.working_memory.get_facts()
                if all_facts:
                    mem = "\n".join(f"- {f.content}" for f in all_facts[:5])
                    memory_context = f"\n\n[PI Memory]:\n{mem}"
                else:
                    memory_context = ""
            except Exception:
                memory_context = ""

            # JSON reminder для малых моделей которые забывают формат
            _JSON_REMINDER = (
                "\n\n[SYSTEM REMINDER: You MUST respond with valid JSON only. "
                "Format: {\"thoughts\":[...],\"tool_name\":\"...\",\"tool_args\":{...}} "
                "No plain text. No markdown. Only JSON object.]"
            )

            # Передаём ВСЕ messages как есть + память + JSON reminder к system
            all_messages = []
            for m in req.messages:
                if m.role == "system":
                    all_messages.append({"role": "system", "content": m.content + memory_context + _JSON_REMINDER})
                else:
                    all_messages.append({"role": m.role, "content": m.content or ""})

            # Generator model: приоритет az_plugin_config → config.yaml → первая из Ollama
            az_cfg = _load_az_plugin_config()
            gen_model = az_cfg.get("generator_model", "").strip() or None
            if not gen_model:
                try:
                    from pure_intellect.engines.config_loader import load_config as _lc
                    gen_model = _lc().generator.model
                except Exception:
                    gen_model = None
            if not gen_model:
                try:
                    _r = httpx.get("http://localhost:11434/api/tags", timeout=5)
                    _models = [m["name"] for m in _r.json().get("models", [])]
                    # Предпочитаем большие модели для генерации (9b > 7b > 4b > 3b > 2b)
                    _preferred = [m for m in _models if any(s in m for s in ["9b","14b","7b","8b"])]
                    # Предпочитаем большие модели для генерации
                    _big_tags = ["32b","30b","27b","24b","14b","9b","8b","7b"]
                    _preferred = [m for m in _models if any(s in m for s in _big_tags)]
                    gen_model = _preferred[0] if _preferred else (_models[0] if _models else None)
                except Exception:
                    gen_model = None
            if not gen_model:
                raise HTTPException(status_code=503, detail="No generator model available. Please configure generator_model in PI Admin Panel or install a model via Ollama.")

            ollama_payload = {
                "model": gen_model,
                "messages": all_messages,
                "temperature": req.temperature,
                "stream": False,
                "options": {"num_ctx": 8192, "num_gpu": -1, "keep_alive": -1},
            }
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(
                    "http://localhost:11434/v1/chat/completions",
                    json=ollama_payload,
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=f"Ollama failed: {resp.text[:200]}")
                data = resp.json()
                # СЫРОЙ ответ — Agent Zero сам парсит JSON и вызывает EXE
                try:
                    raw_content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as _ke:
                    logger.error(f"Ollama response malformed: {_ke} | data={str(data)[:300]}")
                    raise HTTPException(status_code=502, detail=f"Ollama returned unexpected format: {str(data)[:200]}")
                response_text = _extract_first_json(raw_content)

            # Сохраняем факт в память PI
            # ── Coordinate Creation (ModelSwap) ──────────────────────────────
            # Если контекст вырос — создаём координату через coordinator
            _pi_notifications = []
            if len(req.messages) >= _AZ_COORDINATE_MSG_THRESHOLD:
                try:
                    from pure_intellect.utils.swap_manager import get_swap_manager
                    az_cfg2 = _load_az_plugin_config()
                    coord_model = az_cfg2.get("coordinator_model", "").strip()
                    if not coord_model:
                        try:
                            from pure_intellect.engines.config_loader import load_config as _lc2
                            coord_model = _lc2().coordinator.model
                        except Exception:
                            coord_model = ""
                    embed_model = az_cfg2.get("embedding_model", "nomic-embed-text")
                    if coord_model:
                        _pi_notifications.append(
                            "🧠 [Pure Intellect] Контекст заполняется — создаю координату памяти..."
                        )
                        swap = get_swap_manager()
                        await swap.acquire_coordinator(coord_model, embed_model)
                        msg_dicts = [{"role": m.role, "content": m.content or ""} for m in req.messages]
                        coordinate = await _create_az_coordinate(coord_model, msg_dicts)
                        await swap.release_coordinator(coord_model, embed_model)
                        if coordinate:
                            try:
                                from pure_intellect.core.memory.fact import Fact, FactType
                                anchor = Fact(
                                    content=f"[COORDINATE] {coordinate}",
                                    fact_type=FactType.PERMANENT,
                                    is_anchor=True,
                                )
                                pipe.working_memory.add(anchor)
                                logger.info(f"[PI Coordinate] Saved: {coordinate[:80]}")
                                _pi_notifications.append(
                                    f"📍 Координата: {coordinate[:120]}"
                                )
                                _pi_notifications.append(
                                    "✅ [Pure Intellect] Память обновлена — продолжайте работу!"
                                )
                            except Exception as _fe:
                                logger.warning(f"[PI Coordinate] Fact save failed: {_fe}")
                except Exception as _ce:
                    logger.warning(f"[PI Coordinate] Failed: {_ce}")
            # Инжектируем уведомления в thoughts ответа
            if _pi_notifications:
                response_text = _inject_pi_notifications(response_text, _pi_notifications)

            try:
                from pure_intellect.core.memory.fact import Fact, FactType
                fact = Fact(content=f"AgentZero: {query[:100]}", fact_type=FactType.TRANSIENT)
                pipe.working_memory.add(fact)
            except Exception:
                pass

            prompt_tokens = data.get("usage", {}).get("prompt_tokens", len(query.split()) * 2)
            completion_tokens = data.get("usage", {}).get("completion_tokens", len(response_text.split()) * 2)

        else:
            # ── Обычный PI режим (Web UI, прямые API вызовы) ──────────────────
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
            "system_fingerprint": "pure-intellect-v1",
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
        from pure_intellect.utils.hardware_detector import detect_hardware
        return detect_hardware()
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        return {
            "hardware": {"os": "unknown", "ram_gb": 0, "gpu": None},
            "recommendation": {
                "coordinator": "qwen2.5:3b",
                "generator": "qwen2.5:3b",
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

    Body: {"model": "qwen2.5:3b"}
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
                    "http://localhost:11434/api/pull",
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
            resp = await client.get("http://localhost:11434/api/tags")
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


# ── Agent Zero Plugin Config ────────────────────────────────────────────────

_AZ_PLUGIN_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "az_plugin_config.yaml"
)

_DEFAULT_AZ_PLUGIN_CONFIG = {
    "pi_server": "http://host.docker.internal:7860",
    "utility_model": "qwen2.5:3b",
    "generator_model": "",
    "session_id": "agent_zero",
    "recall_threshold": 0.4,
    "recall_limit": 5,
    "recall_enabled": True,
    "memorize_enabled": True,
}


class AZPluginConfigModel(BaseModel):
    pi_server: str = "http://host.docker.internal:7860"
    utility_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"
    generator_model: str = ""
    session_id: str = "agent_zero"
    recall_threshold: float = 0.4
    recall_limit: int = 5
    recall_enabled: bool = True
    memorize_enabled: bool = True

def _load_az_plugin_config() -> dict:
    """Загрузить конфиг плагина AZ из файла."""
    try:
        if os.path.exists(_AZ_PLUGIN_CONFIG_FILE):
            import yaml as _yaml
            with open(_AZ_PLUGIN_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
            return {**_DEFAULT_AZ_PLUGIN_CONFIG, **data}
    except Exception as e:
        logger.warning(f"AZ plugin config load error: {e}")
    return dict(_DEFAULT_AZ_PLUGIN_CONFIG)


def _save_az_plugin_config(config: dict) -> None:
    """Сохранить конфиг плагина AZ в файл."""
    import yaml as _yaml
    with open(_AZ_PLUGIN_CONFIG_FILE, "w", encoding="utf-8") as f:
        _yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


@router.get("/az-plugin/config")
async def get_az_plugin_config():
    """Получить текущий конфиг плагина Agent Zero."""
    config = _load_az_plugin_config()
    # Дополнительно возвращаем список доступных Ollama моделей для dropdown
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            data = resp.json()
            available_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        available_models = []
    return {**config, "available_models": available_models}


@router.post("/az-plugin/config")
async def save_az_plugin_config(config: AZPluginConfigModel):
    """Сохранить конфиг плагина Agent Zero."""
    try:
        config_dict = config.model_dump()
        _save_az_plugin_config(config_dict)
        logger.info(f"AZ plugin config saved: {config_dict}")
        return {"status": "saved", "config": config_dict}
    except Exception as e:
        logger.error(f"AZ plugin config save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
