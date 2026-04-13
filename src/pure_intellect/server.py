"""FastAPI сервер — ядро Оркестратора."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pure_intellect.config import settings
from pure_intellect.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # ─── Startup ───
    print("🧠 Инициализация Оркестратора...")

    # Создаём директории хранения
    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "chromadb").mkdir(exist_ok=True)
    (storage_dir / "archive").mkdir(exist_ok=True)

    # Инициализируем ChromaDB
    import chromadb
    app.state.chroma_client = chromadb.PersistentClient(
        path=str(storage_dir / "chromadb")
    )
    app.state.collection = app.state.chroma_client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"   ✅ ChromaDB: {settings.chroma_collection}")

    # Инициализируем граф
    import networkx as nx
    app.state.graph = nx.DiGraph()
    graph_file = storage_dir / "graph.json"
    if graph_file.exists():
        import json
        data = json.loads(graph_file.read_text())
        app.state.graph = nx.node_link_graph(data)
        print(f"   ✅ Graph: загружен ({app.state.graph.number_of_nodes()} узлов)")
    else:
        print("   ✅ Graph: новый")

    # Инициализируем SQLite для метаданных
    import sqlite3
    db_path = storage_dir / "metadata.db"
    app.state.db = sqlite3.connect(str(db_path))
    app.state.db.execute("""
        CREATE TABLE IF NOT EXISTS file_hashes (
            file_path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            indexed_at REAL NOT NULL,
            status TEXT DEFAULT 'valid'
        )
    """)
    app.state.db.execute("""
        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            tokens_used INTEGER,
            model TEXT,
            rag_hits TEXT
        )
    """)
    app.state.db.commit()
    print("   ✅ SQLite: метаданные")

    print("\n🚀 Оркестратор готов!\n")

    yield

    # ─── Shutdown ───
    print("\n🛑 Завершение работы...")

    # Сохраняем граф
    import json
    graph_data = nx.node_link_data(app.state.graph)
    (storage_dir / "graph.json").write_text(json.dumps(graph_data, indent=2))

    # Закрываем БД
    app.state.db.close()

    print("   ✅ Данные сохранены")


# ─── Создание приложения ───
app = FastAPI(
    title="Чистый Интеллект",
    description="Локальный оркестратор для LLM с иерархической памятью",
    version="0.1.0",
    lifespan=lifespan,
)

# ─── CORS ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───
app.include_router(router)


@app.get("/", tags=["system"])
async def root():
    """Корневой endpoint."""
    return {
        "name": "Чистый Интеллект",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "chat": "POST /chat",
            "index": "POST /index",
            "status": "GET /status",
            "graph": "GET /graph",
            "docs": "GET /docs",
        },
    }
