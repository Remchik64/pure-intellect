"""Pydantic модели для API."""

from typing import Optional, Dict, List, Any
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


class ModelListResponse(BaseModel):
    """Список моделей."""
    models: Dict[str, Any]
