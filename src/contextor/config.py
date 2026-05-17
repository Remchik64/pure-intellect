"""Конфигурация Contextor."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_running_in_docker() -> bool:
    """Определить, запущен ли Contextor внутри Docker-контейнера."""
    # Проверка /.dockerenv
    if Path("/.dockerenv").exists():
        return True
    # Проверка /proc/1/cgroup
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read() or "containerd" in f.read()
    except (FileNotFoundError, PermissionError):
        pass
    return False


def _default_ollama_url() -> str:
    """Вернуть подходящий Ollama URL в зависимости от окружения."""
    if _is_running_in_docker():
        return "http://host.docker.internal:11434"
    return "http://localhost:11434"


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # Server
    host: str = Field(default="0.0.0.0", description="Хост сервера")
    port: int = Field(default=7860, description="Порт сервера")
    debug: bool = Field(default=False, description="Режим отладки")
    
    # Logging
    log_level: str = Field(default="INFO", description="Уровень логирования")
    log_format: str = Field(default="text", description="Формат логов: text или json")
    
    # Models
    orchestrator_model: str = Field(
        default="qwen3.5:2b",
        description="Модель для внутренней логики Оркестратора"
    )
    chat_model: str = Field(
        default="qwen3.5:9b",
        description="Модель для ответов пользователю"
    )
    utility_model: str = Field(
        default="qwen3.5:9b",
        description="Модель для фоновых утилитарных задач и инструментов"
    )
    model_cache_dir: str = Field(
        default="./models",
        description="Директория для кэша моделей"
    )
    gpu_layers: int = Field(
        default=-1,
        description="Количество GPU слоёв (-1 = все)"
    )
    
    # Ollama
    ollama_url: str = Field(
        default_factory=_default_ollama_url,
        description="URL Ollama API (авто: Docker→host.docker.internal, нативно→localhost)"
    )
    ollama_timeout: int = Field(
        default=120,
        description="Таймаут запросов к Ollama (секунды)"
    )
    
    # Storage
    storage_dir: str = Field(default="./storage", description="Директория хранилища")
    chroma_dir: str = Field(default="./storage/chromadb", description="Директория ChromaDB")
    graph_file: str = Field(default="./storage/graph.json", description="Файл графа")
    metadata_db: str = Field(default="./storage/metadata.db", description="SQLite БД")
    
    # LLM parameters
    context_length: int = Field(default=4096, description="Длина контекста")
    temperature: float = Field(default=0.7, description="Температура генерации")
    max_tokens: int = Field(default=2048, description="Максимум токенов в ответе")
    
    # RAG settings
    max_rag_chunks: int = Field(default=5, description="Максимум карточек RAG")
    max_context_tokens: int = Field(default=3000, description="Максимум токенов контекста")
    max_system_prompt_tokens: int = Field(default=2000, description="Максимум токенов system prompt")
    
    model_config = SettingsConfigDict(
        env_prefix="CONTEXTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


def get_settings() -> Settings:
    """Получить настройки."""
    return Settings()

# Глобальный экземпляр настроек
settings = get_settings()
