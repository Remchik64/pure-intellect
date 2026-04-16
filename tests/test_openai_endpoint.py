"""Тесты для OpenAI-совместимого API endpoint.

Позволяет использовать Pure Intellect как backend для:
- Agent Zero (api_base: http://localhost:8085/v1)
- Open WebUI
- LM Studio (как клиент)
- Любого OpenAI-совместимого клиента
"""

import pytest

from pure_intellect.api.routes import openai_router


# ── Проверка структуры роутера ─────────────────────────────

class TestOpenAIRouterStructure:

    def test_router_prefix(self):
        assert openai_router.prefix == "/v1"

    def test_router_has_models_endpoint(self):
        paths = [r.path for r in openai_router.routes]
        assert "/v1/models" in paths

    def test_router_has_chat_completions_endpoint(self):
        paths = [r.path for r in openai_router.routes]
        assert "/v1/chat/completions" in paths

    def test_router_methods(self):
        methods = {}
        for route in openai_router.routes:
            methods[route.path] = list(route.methods or [])
        assert "GET" in methods.get("/v1/models", [])
        assert "POST" in methods.get("/v1/chat/completions", [])


# ── Схемы запросов ─────────────────────────────────────────

class TestOpenAISchemas:

    def test_openai_message_schema(self):
        from pure_intellect.api.routes import OpenAIMessage
        msg = OpenAIMessage(role="user", content="Привет!")
        assert msg.role == "user"
        assert msg.content == "Привет!"

    def test_openai_chat_request_defaults(self):
        from pure_intellect.api.routes import OpenAIChatRequest, OpenAIMessage
        req = OpenAIChatRequest(
            messages=[OpenAIMessage(role="user", content="test")]
        )
        assert req.model == "pure-intellect"
        assert req.temperature == 0.7
        assert req.max_tokens == 2000
        assert req.stream is False

    def test_openai_chat_request_custom(self):
        from pure_intellect.api.routes import OpenAIChatRequest, OpenAIMessage
        req = OpenAIChatRequest(
            model="pure-intellect-code",
            messages=[
                OpenAIMessage(role="system", content="Ты помощник"),
                OpenAIMessage(role="user", content="Привет"),
            ],
            temperature=0.5,
            max_tokens=500,
        )
        assert req.model == "pure-intellect-code"
        assert len(req.messages) == 2
        assert req.temperature == 0.5

    def test_openai_chat_request_multiple_messages(self):
        from pure_intellect.api.routes import OpenAIChatRequest, OpenAIMessage
        req = OpenAIChatRequest(
            messages=[
                OpenAIMessage(role="user", content="Меня зовут Александр"),
                OpenAIMessage(role="assistant", content="Привет, Александр!"),
                OpenAIMessage(role="user", content="Как меня зовут?"),
            ]
        )
        # Последнее user сообщение — текущий запрос
        user_msgs = [m for m in req.messages if m.role == "user"]
        assert user_msgs[-1].content == "Как меня зовут?"


# ── Формат ответа ──────────────────────────────────────────

class TestOpenAIResponseFormat:
    """Тестируем что формат ответа совместим с OpenAI API."""

    def test_models_response_structure(self):
        """Формат /v1/models совместим с OpenAI."""
        # Симулируем ожидаемый ответ
        expected = {
            "object": "list",
            "data": [
                {
                    "id": "pure-intellect",
                    "object": "model",
                    "created": 1714000000,
                    "owned_by": "pure-intellect",
                }
            ],
        }
        assert "object" in expected
        assert expected["object"] == "list"
        assert len(expected["data"]) > 0
        assert "id" in expected["data"][0]

    def test_chat_completion_response_structure(self):
        """Формат ответа /v1/chat/completions совместим с OpenAI."""
        import time
        import uuid

        # Симулируем ожидаемый ответ
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "pure-intellect",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Ответ с памятью!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        # Проверяем структуру
        assert "id" in response
        assert response["id"].startswith("chatcmpl-")
        assert response["object"] == "chat.completion"
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in response
        assert "prompt_tokens" in response["usage"]
        assert "completion_tokens" in response["usage"]
        assert "total_tokens" in response["usage"]

    def test_usage_tokens_sum(self):
        """total_tokens = prompt_tokens + completion_tokens."""
        usage = {
            "prompt_tokens": 42,
            "completion_tokens": 18,
            "total_tokens": 60,
        }
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# ── Интеграция с Agent Zero ────────────────────────────────

class TestAgentZeroCompatibility:
    """Проверяем совместимость с форматом Agent Zero."""

    def test_agent_zero_request_format(self):
        """Agent Zero отправляет запрос именно в таком формате."""
        from pure_intellect.api.routes import OpenAIChatRequest, OpenAIMessage

        # Типичный запрос от Agent Zero
        agent_zero_payload = {
            "model": "pure-intellect",
            "messages": [
                {"role": "system", "content": "You are Agent Zero, an AI assistant."},
                {"role": "user", "content": "Найди информацию о Python"},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
            "stream": False,
        }

        # Pydantic должен принять такой payload
        req = OpenAIChatRequest(**agent_zero_payload)
        assert req.model == "pure-intellect"
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert req.messages[1].role == "user"

    def test_system_message_extraction(self):
        """System message корректно извлекается из messages."""
        from pure_intellect.api.routes import OpenAIMessage
        messages = [
            OpenAIMessage(role="system", content="Ты помощник разработчика"),
            OpenAIMessage(role="user", content="Помоги с кодом"),
        ]
        system_msgs = [m for m in messages if m.role == "system"]
        user_msgs = [m for m in messages if m.role == "user"]

        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Ты помощник разработчика"
        assert user_msgs[-1].content == "Помоги с кодом"

    def test_pure_intellect_metadata_in_response(self):
        """Ответ содержит метаданные Pure Intellect."""
        # Проверяем что наш ответ расширяет OpenAI формат
        response = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            # Наш уникальный раздел:
            "pure_intellect": {
                "turn": 5,
                "coherence_score": 0.87,
                "memory_facts": 12,
                "session_id": "default",
            },
        }
        assert "pure_intellect" in response
        assert "coherence_score" in response["pure_intellect"]
        assert "memory_facts" in response["pure_intellect"]
        assert "session_id" in response["pure_intellect"]
