"""API routes для Оркестратора."""

import time
import json

import httpx
from fastapi import APIRouter, Request, HTTPException

from pure_intellect.config import settings
from pure_intellect.api.schemas import (
    ChatRequest,
    ChatResponse,
    IndexRequest,
    IndexResponse,
    StatusResponse,
    GraphResponse,
    GraphNode,
)

router = APIRouter()


# ═══════════════════════════════════════════════════════
#  CHAT — Основной endpoint
# ═══════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: Request, body: ChatRequest):
    """
    Отправить запрос через Оркестратор.
    
    Оркестратор:
    1. Ищет релевантный контекст (RAG + Graph)
    2. Собирает system prompt с карточками кода
    3. Отправляет в Ollama
    4. Логирует ответ
    """
    start_time = time.time()
    
    # Определяем модель
    model = body.model or settings.default_model
    
    # ─── Шаг 1: RAG поиск ───
    rag_chunks = []
    rag_hits = []
    
    collection = request.app.state.collection
    try:
        results = collection.query(
            query_texts=[body.query],
            n_results=settings.max_rag_chunks,
        )
        if results and results["documents"] and results["documents"][0]:
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0])
            ):
                rag_chunks.append(doc)
                entity_name = meta.get("entity_name", "unknown")
                rag_hits.append(entity_name)
    except Exception as e:
        # Если коллекция пуста — это нормально
        pass
    
    # ─── Шаг 2: Поиск в графе ───
    graph_context = ""
    graph = request.app.state.graph
    if graph.number_of_nodes() > 0:
        # Ищем узлы, связанные с запросом
        query_lower = body.query.lower()
        related_nodes = []
        for node_id, data in graph.nodes(data=True):
            name = data.get("name", "").lower()
            summary = data.get("summary", "").lower()
            if any(word in name or word in summary for word in query_lower.split()):
                related_nodes.append(data)
        
        if related_nodes:
            graph_context = "\n".join([
                f"- {n.get('name', '?')} ({n.get('type', '?')}): {n.get('summary', '')}"
                for n in related_nodes[:5]
            ])
    
    # ─── Шаг 3: Сборка system prompt ───
    system_prompt = _build_system_prompt(
        mode=body.mode.value,
        rag_chunks=rag_chunks,
        graph_context=graph_context,
    )
    
    # ─── Шаг 4: Отправка в Ollama ───
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": body.query},
    ]
    
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            response = await client.post(
                f"{settings.ollama_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": body.temperature,
                    "max_tokens": body.max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama недоступен по адресу {settings.ollama_url}. Убедитесь, что Ollama запущена.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка Ollama: {e.response.status_code} — {e.response.text}",
        )
    
    # ─── Шаг 5: Логирование ───
    response_time = time.time() - start_time
    
    db = request.app.state.db
    db.execute(
        "INSERT INTO conversation_log (timestamp, query, response, tokens_used, model, rag_hits) VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), body.query, answer, tokens_used, model, json.dumps(rag_hits)),
    )
    db.commit()
    
    return ChatResponse(
        response=answer,
        model=model,
        tokens_used=tokens_used,
        rag_hits=rag_hits,
        context_tokens=len(system_prompt.split()),
        response_time=round(response_time, 2),
    )


# ═══════════════════════════════════════════════════════
#  INDEX — Индексация проекта
# ═══════════════════════════════════════════════════════

@router.post("/index", response_model=IndexResponse, tags=["index"])
async def index_project(request: Request, body: IndexRequest):
    """
    Проиндексировать проект.
    
    1. Сканирует файлы
    2. Парсит через tree-sitter
    3. Генерирует карточки сущностей
    4. Создаёт эмбеддинги → ChromaDB
    5. Строит граф связей
    """
    from pathlib import Path
    import hashlib
    
    project_path = Path(body.path)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Путь не найден: {body.path}")
    
    start_time = time.time()
    files_indexed = 0
    entities_found = 0
    chunks_created = 0
    
    collection = request.app.state.collection
    graph = request.app.state.graph
    db = request.app.state.db
    
    # Сканируем файлы
    for file_path in project_path.rglob("*"):
        # Фильтрация
        if not file_path.is_file():
            continue
        if file_path.suffix not in settings.supported_extensions:
            continue
        if any(ignored in file_path.parts for ignored in settings.ignore_dirs):
            continue
        if any(file_path.name.endswith(ignored) for ignored in settings.ignore_files):
            continue
        
        # Проверяем хэш (ленивая переиндексация)
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Проверяем, изменился ли файл
        if not body.force:
            cursor = db.execute(
                "SELECT sha256 FROM file_hashes WHERE file_path = ?",
                (str(file_path),),
            )
            row = cursor.fetchone()
            if row and row[0] == file_hash:
                continue  # Файл не изменился
        
        # Парсим файл
        entities = _parse_file(file_path, content)
        
        for entity in entities:
            # Создаём карточку
            card_text = _entity_to_card(entity, str(file_path))
            
            # Добавляем в ChromaDB
            doc_id = f"{file_path}:{entity['name']}"
            collection.upsert(
                ids=[doc_id],
                documents=[card_text],
                metadatas=[{
                    "file": str(file_path),
                    "entity_name": entity["name"],
                    "entity_type": entity["type"],
                    "project": body.path,
                }],
            )
            chunks_created += 1
            
            # Добавляем в граф
            graph.add_node(
                doc_id,
                name=entity["name"],
                type=entity["type"],
                file=str(file_path),
                summary=entity.get("summary", ""),
            )
            
            # Добавляем связи (imports, calls)
            for dep in entity.get("calls", []):
                graph.add_edge(doc_id, dep)
            
            entities_found += 1
        
        # Обновляем хэш
        db.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, sha256, indexed_at) VALUES (?, ?, ?)",
            (str(file_path), file_hash, time.time()),
        )
        db.commit()
        files_indexed += 1
    
    duration = time.time() - start_time
    
    return IndexResponse(
        status="success",
        files_indexed=files_indexed,
        entities_found=entities_found,
        chunks_created=chunks_created,
        duration=round(duration, 2),
    )


# ═══════════════════════════════════════════════════════
#  STATUS / GRAPH
# ═══════════════════════════════════════════════════════

@router.get("/status", response_model=StatusResponse, tags=["system"])
async def status(request: Request):
    """Статус системы."""
    import httpx
    
    # Проверяем Ollama
    ollama_connected = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            ollama_connected = resp.status_code == 200
    except Exception:
        pass
    
    # Счётчики
    db = request.app.state.db
    cursor = db.execute("SELECT COUNT(*) FROM file_hashes")
    files_indexed = cursor.fetchone()[0]
    
    cursor = db.execute("SELECT COUNT(*) FROM conversation_log")
    conversations = cursor.fetchone()[0]
    
    return StatusResponse(
        status="running",
        ollama_connected=ollama_connected,
        model=settings.default_model,
        nodes_in_graph=request.app.state.graph.number_of_nodes(),
        files_indexed=files_indexed,
        conversations_logged=conversations,
    )


@router.get("/graph", response_model=GraphResponse, tags=["system"])
async def get_graph(request: Request):
    """Получить граф знаний."""
    graph = request.app.state.graph
    
    nodes = [
        GraphNode(
            id=node_id,
            type=data.get("type", "unknown"),
            name=data.get("name", node_id),
            file=data.get("file", ""),
            summary=data.get("summary", ""),
        )
        for node_id, data in graph.nodes(data=True)
    ]
    
    edges = list(graph.edges())
    
    return GraphResponse(
        nodes=nodes,
        edges=edges,
        total_nodes=len(nodes),
        total_edges=len(edges),
    )


# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════

def _build_system_prompt(
    mode: str,
    rag_chunks: list[str],
    graph_context: str,
) -> str:
    """Собрать system prompt из контекста."""
    
    mode_instructions = {
        "analyze": "Ты Senior разработчик. Анализируй код, находи ошибки, предлагай решения.",
        "code": "Ты Senior разработчик. Пиши чистый, production-ready код с типами и docstrings.",
        "explain": "Ты Senior разработчик. Объясняй код простым языком, разбирай по шагам.",
        "refactor": "Ты Senior разработчик. Предлагай рефакторинг: SOLID, DRY, читаемость.",
        "chat": "Ты Senior разработчик. Отвечай на основе предоставленного контекста.",
    }
    
    parts = [mode_instructions.get(mode, mode_instructions["chat"])]
    
    if graph_context:
        parts.append(f"\n[АРХИТЕКТУРА ПРОЕКТА]\n{graph_context}")
    
    if rag_chunks:
        cards = "\n\n".join(rag_chunks)
        parts.append(f"\n[КОД ИЗ ПРОЕКТА]\n{cards}")
    
    parts.append("\nОтвечай строго на основе предоставленного контекста.")
    parts.append("Если информации недостаточно — честно скажи, что нужно больше контекста.")
    
    return "\n".join(parts)


def _parse_file(file_path, content: str) -> list[dict]:
    """Парсинг файла через tree-sitter."""
    # Заглушка — полная реализация в parsers/
    entities = []
    
    if file_path.suffix == ".py":
        # Простой regex fallback для MVP
        import re
        
        # Ищем классы
        for match in re.finditer(r"^class\s+(\w+)(?:\([^)]*\))?\s*:", content, re.MULTILINE):
            entities.append({
                "name": match.group(1),
                "type": "class",
                "summary": f"Class {match.group(1)}",
                "calls": [],
            })
        
        # Ищем функции
        for match in re.finditer(r"^def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?\s*:", content, re.MULTILINE):
            entities.append({
                "name": match.group(1),
                "type": "function",
                "summary": f"Function {match.group(1)}",
                "calls": [],
            })
    
    elif file_path.suffix in (".js", ".ts", ".jsx", ".tsx"):
        import re
        
        for match in re.finditer(r"(?:function|const|let|var)\s+(\w+)", content):
            entities.append({
                "name": match.group(1),
                "type": "function",
                "summary": f"Function/variable {match.group(1)}",
                "calls": [],
            })
    
    return entities


def _entity_to_card(entity: dict, file_path: str) -> str:
    """Преобразовать сущность в текстовую карточку."""
    lines = [
        f"---",
        f"card: {entity['name']}",
        f"type: {entity['type']}",
        f"file: {file_path}",
        f"summary: {entity.get('summary', '')}",
    ]
    
    if entity.get("calls"):
        lines.append(f"calls: [{', '.join(entity['calls'])}]")
    
    lines.append(f"---")
    return "\n".join(lines)
