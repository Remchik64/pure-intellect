"""Ollama engine — клиент для локальной Ollama."""

import time
import httpx

from pure_intellect.engines.base import BaseEngine, LLMResponse
from pure_intellect.config import settings


class OllamaEngine(BaseEngine):
    """Движок Ollama."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.ollama_url
        self.timeout = settings.ollama_timeout

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Отправить запрос в Ollama."""
        model = model or settings.default_model
        start = time.time()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            response_time=round(time.time() - start, 2),
        )

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Стриминг ответа из Ollama."""
        model = model or settings.default_model
        collected = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            import json
                            parsed = json.loads(chunk)
                            delta = parsed["choices"][0].get("delta", {})
                            if "content" in delta:
                                collected.append(delta["content"])
                        except Exception:
                            continue

        return "".join(collected)

    async def is_available(self) -> bool:
        """Проверить доступность Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Получить список скачанных моделей."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def pull_model(self, model_name: str) -> bool:
        """Скачать модель."""
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": False},
                )
                resp.raise_for_status()
                return True
        except Exception:
            return False
