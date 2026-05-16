"""Pydantic модели для API."""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Запрос на чат."""
    query: str = Field(..., description="Запрос пользователя")
    model: Optional[str] = Field(default=None, description="Ключ модели")
    system: Optional[str] = Field(default=None, description="System prompt")
    temperature: Optional[float] = Field(default=0.7, description="Температура")
    project: Optional[str] = Field(default=None, description="Имя проекта")


class ChatResponse(BaseModel):
    """Ответ чата."""
    response: str = Field(..., description="Ответ модели")
    model: str = Field(..., description="Использованная модель")


class HealthResponse(BaseModel):
    """Ответ здоровья."""
    status: str
    model_loaded: bool
    version: str


class SwitchModelRequest(BaseModel):
    """Запрос на переключение модели."""
    role: str = Field(..., description="Ключ модели: coordinator или generator")
    model: str = Field(..., description="Имя модели (например, qwen3.5:9b)")


class DownloadModelRequest(BaseModel):
    """Запрос на скачивание модели."""
    model: str = Field(..., description="Имя модели для скачивания")


class CreateSessionRequest(BaseModel):
    """Запрос на создание новой сессии."""
    display_name: Optional[str] = Field(default=None, description="Имя чата (default: 'New Chat')")
    session_type: Optional[str] = Field(default="chat", description="Тип сессии: chat или project")


class RenameSessionRequest(BaseModel):
    """Запрос на переименование сессии."""
    display_name: str = Field(..., description="Новое имя чата")
