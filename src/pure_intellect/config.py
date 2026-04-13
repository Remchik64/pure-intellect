"""Конфигурация проекта Чистый Интеллект."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Настройки приложения."""

    # ─── Сервер ───
    host: str = Field(default="0.0.0.0", description="Хост сервера")
    port: int = Field(default=8085, description="Порт сервера")
    debug: bool = Field(default=False, description="Режим отладки")

    # ─── Ollama ───
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="URL Ollama сервера",
    )
    default_model: str = Field(
        default="qwen2.5:7b",
        description="Модель по умолчанию",
    )
    ollama_timeout: int = Field(
        default=120,
        description="Таймаут запроса к Ollama (секунды)",
    )

    # ─── Контекст ───
    max_context_tokens: int = Field(
        default=8192,
        description="Максимальный размер контекстного окна",
    )
    max_system_prompt_tokens: int = Field(
        default=3000,
        description="Максимум токенов на system prompt",
    )
    max_rag_chunks: int = Field(
        default=5,
        description="Максимум RAG-чанков в контексте",
    )
    max_unfold_depth: int = Field(
        default=2,
        description="Максимум распаковок (lazy unpacking)",
    )

    # ─── Хранилище ───
    storage_dir: Path = Field(
        default=Path("./storage"),
        description="Директория хранения данных",
    )
    chroma_collection: str = Field(
        default="code_cards",
        description="Имя коллекции ChromaDB",
    )

    # ─── Индексация ───
    supported_extensions: list[str] = Field(
        default=[".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb"],
        description="Поддерживаемые расширения файлов",
    )
    ignore_dirs: list[str] = Field(
        default=[
            ".git",
            "node_modules",
            "venv",
            "__pycache__",
            ".venv",
            "dist",
            "build",
            ".tox",
            ".mypy_cache",
            ".ruff_cache",
        ],
        description="Игнорируемые директории",
    )
    ignore_files: list[str] = Field(
        default=[".pyc", ".pyo", ".log", ".min.js", ".min.css", ".map"],
        description="Игнорируемые расширения файлов",
    )

    # ─── Логирование ───
    log_level: str = Field(default="INFO", description="Уровень логирования")
    log_format: str = Field(
        default="json",
        description="Формат логов (json или pretty)",
    )

    class Config:
        env_prefix = "PI_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
