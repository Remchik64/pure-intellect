"""WebSocket стриминг для реал-тайм ответов LLM."""

import json
import logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from ..engine import ModelManager, MODEL_REGISTRY

logger = logging.getLogger(__name__)


class StreamingManager:
    """Управление WebSocket соединениями и стримингом."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.model_manager: Optional[ModelManager] = None
    
    def get_model_manager(self) -> ModelManager:
        """Получить или создать ModelManager."""
        if self.model_manager is None:
            self.model_manager = ModelManager(cache_dir="./models")
        return self.model_manager
    
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
        """Стримить ответ LLM токен за токеном."""
        manager = self.get_model_manager()
        
        # Параметры запроса
        messages = data.get("messages", [])
        model_key = data.get("model", "qwen2.5-coder-7b")
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 2048)
        
        # Если нет messages, формируем из query/system
        if not messages:
            query = data.get("query", "")
            system = data.get("system")
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": query})
        
        # Загрузить модель если не загружена
        if manager.loaded_model is None:
            if model_key not in MODEL_REGISTRY:
                await self.send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown model: {model_key}"
                })
                return
            
            await self.send_json(websocket, {
                "type": "status",
                "message": f"Loading model {model_key}..."
            })
            
            try:
                manager.load(model_key, n_gpu_layers=-1)
            except Exception as e:
                await self.send_json(websocket, {
                    "type": "error",
                    "message": f"Failed to load model: {e}"
                })
                return
        
        # Стриминг генерации
        await self.send_json(websocket, {
            "type": "start",
            "model": model_key,
        })
        
        try:
            # llama-cpp-python streaming
            stream = manager.loaded_model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,  # КЛЮЧЕВОЙ ПАРАМЕТР!
            )
            
            full_response = ""
            token_count = 0
            
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    full_response += content
                    token_count += 1
                    
                    # Отправляем токен клиенту
                    await self.send_json(websocket, {
                        "type": "token",
                        "content": content,
                    })
            
            # Завершение стриминга
            await self.send_json(websocket, {
                "type": "end",
                "full_response": full_response,
                "tokens": token_count,
            })
            
            logger.info(f"Stream complete: {token_count} tokens")
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            await self.send_json(websocket, {
                "type": "error",
                "message": str(e)
            })


# Глобальный менеджер
streaming_manager = StreamingManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint для стриминга."""
    await streaming_manager.connect(websocket)
    
    try:
        while True:
            # Ждём сообщение от клиента
            data = await websocket.receive_json()
            
            action = data.get("action", "chat")
            
            if action == "chat":
                await streaming_manager.stream_chat(websocket, data)
            elif action == "ping":
                await streaming_manager.send_json(websocket, {"type": "pong"})
            elif action == "status":
                manager = streaming_manager.get_model_manager()
                await streaming_manager.send_json(websocket, {
                    "type": "status",
                    "model_loaded": manager.loaded_model is not None,
                    "connections": len(streaming_manager.active_connections),
                })
            else:
                await streaming_manager.send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown action: {action}"
                })
    
    except WebSocketDisconnect:
        streaming_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        streaming_manager.disconnect(websocket)
