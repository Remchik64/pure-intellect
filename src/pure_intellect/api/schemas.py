"""Pydantic модели для API endpoints."""

from enum import Enum
from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Режимы работы."""
    ANALYZE = "analyze"       # Анализ кода / отладка
    CODE = "code"             # Генерация кода
    EXPLAIN = "explain"       # Объяснение
    REFACTOR = "refactor"     # Рефакторинг
    CHAT = "chat"             # Свободный чат


class ChatRequest(BaseModel):
    """Запрос на чат."""
    query: str = Field(..., description="Запрос пользователя")
    project: str = Field(default="default", description="Имя проекта")
    model: str | None = Field(default=None, description="Модель (None = default)")
    mode: Mode = Field(default=Mode.CHAT, description="Режим работы")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    stream: bool = Field(default=False, description="Стриминг ответа")


class IndexRequest(BaseModel):
    """Запрос на индексацию."""
    path: str = Field(..., description="Путь к директории проекта")
    force: bool = Field(default=False, description="Принудительная переиндексация")


class ChatResponse(BaseModel):
    """Ответ на чат."""
    response: str = Field(..., description="Ответ модели")
    model: str = Field(..., description="Использованная модель")
    tokens_used: int = Field(default=0, description="Потрачено токенов")
    rag_hits: list[str] = Field(default_factory=list, description="Найденные карточки")
    context_tokens: int = Field(default=0, description="Токенов в контексте")
    response_time: float = Field(default=0.0, description="Время ответа (сек)")


class IndexResponse(BaseModel):
    """Ответ на индексацию."""
    status: str = Field(..., description="Статус операции")
    files_indexed: int = Field(default=0, description="Проиндексировано файлов")
    entities_found: int = Field(default=0, description="Найдено сущностей")
    chunks_created: int = Field(default=0, description="Создано чанков")
    duration: float = Field(default=0.0, description="Время индексации (сек)")


class StatusResponse(BaseModel):
    """Статус системы."""
    status: str = Field(..., description="Статус")
    ollama_connected: bool = Field(..., description="Ollama доступен")
    model: str = Field(..., description="Активная модель")
    nodes_in_graph: int = Field(default=0, description="Узлов в графе")
    files_indexed: int = Field(default=0, description="Файлов в индексе")
    conversations_logged: int = Field(default=0, description="Записей в архиве")


class GraphNode(BaseModel):
    """Узел графа знаний."""
    id: str = Field(..., description="Идентификатор узла")
    type: str = Field(..., description="Тип: function, class, module")
    name: str = Field(..., description="Имя сущности")
    file: str = Field(default="", description="Файл")
    summary: str = Field(default="", description="Краткое описание")


class GraphResponse(BaseModel):
    """Ответ с графом знаний."""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    total_nodes: int = Field(default=0)
    total_edges: int = Field(default=0)
