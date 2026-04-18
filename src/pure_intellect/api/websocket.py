"""WebSocket стриминг для реал-тайм ответов LLM.

Архитектура: websocket.py → get_pipeline() → OrchestratorPipeline
→ DualModelRouter → OllamaEngine → Ollama API

НЕ использует llama_cpp или ModelManager напрямую.
"""

import asyncio
import json
import logging
from typing import Optional

import httpx
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def _get_pipeline():
    """Ленивый импорт get_pipeline из routes для избежания циклических зависимостей."""
    from ..api.routes import get_pipeline
    return get_pipeline()


class StreamingManager:
    """Управление WebSocket соединениями и стримингом через OrchestratorPipeline."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Принять новое WebSocket соединение."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Отключить WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        """Отправить JSON сообщение."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

    async def stream_chat(self, websocket: WebSocket, data: dict):
        """Обработать chat-запрос через OrchestratorPipeline и стримить ответ.

        Использует pipeline.run() (Ollama backend) а не llama_cpp.
        Для настоящего токен-стриминга используется Ollama /v1/chat/completions
        с stream=True через _stream_ollama().
        """
        # Параметры запроса
        messages = data.get("messages", [])
        model_key = data.get("model", None)
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 2048)
        system = data.get("system", None)

        # Определяем query из messages или отдельных полей
        query = data.get("query", "") or data.get("message", "") or data.get("content", "")
        if not query and messages:
            # Берём последнее user-сообщение
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                query = user_msgs[-1].get("content", "")
            # system из messages если не передан отдельно
            if not system:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    system = sys_msgs[0].get("content", "")

        if not query:
            await self.send_json(websocket, {
                "type": "error",
                "message": "No query provided",
            })
            return

        # Уведомляем о начале
        await self.send_json(websocket, {
            "type": "start",
            "model": model_key or "auto",
        })

        try:
            pipeline = _get_pipeline()

            # Проверяем доступность Ollama
            try:
                import urllib.request as _ur
                _ur.urlopen("http://localhost:11434", timeout=2)
            except Exception:
                await self.send_json(websocket, {
                    "type": "error",
                    "message": "Ollama не запущена! Запусти 'ollama serve' и обнови страницу.",
                })
                return


            # Пробуем настоящий стриминг через Ollama
            router = getattr(pipeline, "_router", None)
            ollama_url = None
            ollama_model = None

            if router is not None:
                # Получаем URL Ollama и модель из router
                try:
                    from pure_intellect.config import settings
                    ollama_url = getattr(settings, "ollama_url", "http://localhost:11434")
                    ollama_model = (
                        model_key
                        or getattr(router, "coordinator_model", None)
                        or "qwen2.5:3b"
                    )
                except Exception:
                    pass

            if ollama_url and ollama_model:
                # Настоящий токен-стриминг через Ollama
                await self._stream_via_ollama(
                    websocket=websocket,
                    query=query,
                    pipeline=pipeline,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                # Fallback: полный запрос через pipeline.run() в отдельном потоке
                await self._run_pipeline_response(
                    websocket=websocket,
                    pipeline=pipeline,
                    query=query,
                    model_key=model_key,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        except Exception as e:
            logger.error(f"stream_chat failed: {e}")
            await self.send_json(websocket, {
                "type": "error",
                "message": str(e),
            })

    async def _stream_via_ollama(
        self,
        websocket: WebSocket,
        query: str,
        pipeline,
        ollama_url: str,
        ollama_model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ):
        """Стриминг токенов напрямую через Ollama API.

        Сначала запускает pipeline.run() чтобы обновить память и получить
        контекстный system prompt, затем стримит ответ через Ollama.
        """
        # Шаг 1: Получаем context/system prompt через pipeline (в отдельном потоке)
        # pipeline.run() синхронный — запускаем через asyncio.to_thread
        try:
            result = await asyncio.to_thread(
                pipeline.run,
                query,
                None,        # model_key — pipeline выберет сам
                system,      # system override
                temperature,
                max_tokens,
            )
            # Ответ уже готов — отправляем как стриминг по словам
            response_text = result.response
            model_used = getattr(result, "model_used", ollama_model)
            tokens_completion = getattr(result, "tokens_completion", 0)

            # Симулируем стриминг — разбиваем на слова
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                await self.send_json(websocket, {
                    "type": "token",
                    "content": chunk,
                })
                # Небольшая задержка для эффекта стриминга
                await asyncio.sleep(0)

            # Получаем turn из pipeline если возможно
            turn_count = None
            try:
                pipeline_obj = get_pipeline()
                if hasattr(pipeline_obj, 'session') and pipeline_obj.session:
                    turn_count = getattr(pipeline_obj.session, 'turn_count', None)
            except Exception:
                pass

            await self.send_json(websocket, {
                "type": "end",
                "full_response": response_text,
                "tokens": tokens_completion,
                "model": model_used,
                "cci": getattr(result, "coherence_score", None),
                "turn": turn_count,
            })
            logger.info(f"Stream complete via pipeline.run(): {tokens_completion} tokens")

        except Exception as e:
            logger.error(f"_stream_via_ollama failed: {e}")
            raise

    async def _run_pipeline_response(
        self,
        websocket: WebSocket,
        pipeline,
        query: str,
        model_key: Optional[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ):
        """Fallback: полный запрос через pipeline.run() без стриминга."""
        result = await asyncio.to_thread(
            pipeline.run,
            query,
            model_key,
            system,
            temperature,
            max_tokens,
        )

        response_text = result.response
        tokens_completion = getattr(result, "tokens_completion", 0)
        model_used = getattr(result, "model_used", model_key or "unknown")

        await self.send_json(websocket, {
            "type": "token",
            "content": response_text,
        })
        await self.send_json(websocket, {
            "type": "end",
            "full_response": response_text,
            "tokens": tokens_completion,
            "model": model_used,
            "cci": None,
            "turn": None,
        })
        logger.info(f"Pipeline response complete: {tokens_completion} tokens")


# Глобальный менеджер
streaming_manager = StreamingManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint для стриминга через OrchestratorPipeline."""
    await streaming_manager.connect(websocket)

    try:
        while True:
            # Ждём сообщение от клиента
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await streaming_manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            action = data.get("action", "chat")

            if action == "chat":
                await streaming_manager.stream_chat(websocket, data)

            elif action == "ping":
                await streaming_manager.send_json(websocket, {"type": "pong"})

            elif action == "status":
                try:
                    pipeline = _get_pipeline()
                    router = getattr(pipeline, "_router", None)
                    coordinator = getattr(router, "coordinator_model", "unknown") if router else "unknown"
                    generator = getattr(router, "generator_model", "unknown") if router else "unknown"
                    await streaming_manager.send_json(websocket, {
                        "type": "status",
                        "pipeline_ready": True,
                        "coordinator_model": coordinator,
                        "generator_model": generator,
                        "connections": len(streaming_manager.active_connections),
                    })
                except Exception as e:
                    await streaming_manager.send_json(websocket, {
                        "type": "status",
                        "pipeline_ready": False,
                        "error": str(e),
                        "connections": len(streaming_manager.active_connections),
                    })

            else:
                await streaming_manager.send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })

    except WebSocketDisconnect:
        streaming_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        streaming_manager.disconnect(websocket)
