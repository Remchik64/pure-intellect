"""Движок для работы с локальными LLM моделями."""

from .model_manager import ModelManager
from .registry import MODEL_REGISTRY

__all__ = ["ModelManager", "MODEL_REGISTRY"]
