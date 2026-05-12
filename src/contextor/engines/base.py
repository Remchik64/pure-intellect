"""Базовый абстрактный класс для LLM движков."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Ответ от LLM движка."""
    content: str
    model: str
    tokens_used: int = 0
    response_time: float = 0.0


class BaseEngine(ABC):
    """Абстрактный LLM движок."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Отправить запрос и получить ответ."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Проверить доступность движка."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Получить список доступных моделей."""
        ...
