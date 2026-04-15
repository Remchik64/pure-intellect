"""API endpoints."""

import logging
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException
from ..api.schemas import ChatRequest, ChatResponse, HealthResponse, ModelListResponse
from ..engine import ModelManager, MODEL_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter()

def get_model_manager() -> ModelManager:
    """Получить thread-safe singleton ModelManager."""
    return ModelManager.get_instance(cache_dir="./models")


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
async def orchestrate(
    query: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    use_llm_intent: bool = False,
):
    """Полный пайплайн: Intent → RAG → Graph → Assembler → LLM."""
    from ..core import OrchestratorPipeline
    
    try:
        manager = get_model_manager()
        pipeline = OrchestratorPipeline(model_manager=manager)
        
        result = pipeline.run(
            query=query,
            model_key=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            use_llm_intent=use_llm_intent,
        )
        
        return result.to_dict()
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
