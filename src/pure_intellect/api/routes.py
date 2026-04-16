"""API endpoints."""

import logging
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..api.schemas import ChatRequest, ChatResponse, HealthResponse, ModelListResponse, OrchestrateRequest
from ..engine import ModelManager, MODEL_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton pipeline — сохраняет память между запросами
_pipeline = None
import threading
_pipeline_lock = threading.Lock()


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
        return pipeline.get_sessions()
    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/new")
async def create_new_session(req: NewSessionRequest):
    """Создать новую сессию и переключиться на неё."""
    try:
        result = pipeline.create_new_session(
            display_name=req.display_name,
            session_type=req.session_type,
            project_path=req.project_path,
        )
        return result
    except Exception as e:
        logger.error(f"Create session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/switch")
async def switch_session(session_id: str):
    """Переключить активную сессию."""
    try:
        result = pipeline.switch_session(session_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Session not found"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Switch session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}/rename")
async def rename_session(session_id: str, req: RenameSessionRequest):
    """Переименовать сессию."""
    try:
        result = pipeline.rename_session(session_id, req.display_name)
        return result
    except Exception as e:
        logger.error(f"Rename session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Удалить сессию."""
    try:
        result = pipeline.delete_session_by_id(session_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail="Cannot delete this session")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session failed: {e}")
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


# ── OpenAI-Compatible API (для Agent Zero, Open WebUI, LM Studio) ─────
# Позволяет использовать Pure Intellect как OpenAI-совместимый сервер
# Agent Zero config: api_base = http://localhost:8085/v1

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

    Принимает запросы в формате OpenAI API и возвращает ответы
    с полной поддержкой иерархической памяти Pure Intellect.

    Настройка Agent Zero:
      api_base: http://localhost:8085/v1
      api_key: pure-intellect  (любой)
      model: pure-intellect
    """
    import time
    import uuid

    try:
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

        # Прогоняем через OrchestratorPipeline с ПАМЯТЬЮ
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
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
            # Дополнительная мета-информация Pure Intellect
            "pure_intellect": {
                "turn": result.intent.turn if hasattr(result.intent, "turn") else 0,
                "coherence_score": result.coherence_score,
                "memory_facts": pipe.working_memory.size(),
                "session_id": pipe._session_manager.active_session_id,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
