"""Конфигурация Чистый Интеллект."""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # Server
    host: str = Field(default="0.0.0.0", description="Хост сервера")
    port: int = Field(default=8085, description="Порт сервера")
    debug: bool = Field(default=False, description="Режим отладки")
    
    # Logging
    log_level: str = Field(default="INFO", description="Уровень логирования")
    log_format: str = Field(default="text", description="Формат логов: text или json")
    
    # Models
    orchestrator_model: str = Field(
        default="qwen2.5-3b",
        description="Модель для внутренней логики Оркестратора"
    )
    chat_model: str = Field(
        default="qwen2.5-coder-7b",
        description="Модель для ответов пользователю"
    )
    model_cache_dir: str = Field(
        default="./models",
        description="Директория для кэша моделей"
    )
    gpu_layers: int = Field(
        default=-1,
        description="Количество GPU слоёв (-1 = все)"
    )
    
    # Storage
    storage_dir: str = Field(default="./storage", description="Директория хранилища")
    chroma_dir: str = Field(default="./storage/chromadb", description="Директория ChromaDB")
    graph_file: str = Field(default="./storage/graph.json", description="Файл графа")
    metadata_db: str = Field(default="./storage/metadata.db", description="SQLite БД")
    archive_dir: str = Field(default="./storage/archive", description="Директория архива")
    
    # LLM parameters
    context_length: int = Field(default=4096, description="Длина контекста")
    temperature: float = Field(default=0.7, description="Температура генерации")
    max_tokens: int = Field(default=2048, description="Максимум токенов в ответе")
    
    class Config:
        env_prefix = "PURE_INTELLECT_"
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Получить настройки."""
    return Settings()

# Глобальный экземпляр настроек
settings = get_settings()
