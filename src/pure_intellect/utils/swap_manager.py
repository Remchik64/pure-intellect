"""ModelSwapManager — умное управление моделями в VRAM.

Логика:
  Обычный режим:   generator (большой) + embedding в VRAM
  При координации: выгрузить embedding
                   → загрузить coordinator (маленький)
                   → создать координату/слепок
                   → выгрузить coordinator
                   → загрузить embedding обратно

Это позволяет держать большую модель (20B+) в VRAM постоянно,
а coordinator запускать только когда нужно (5% времени).
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_OLLAMA_BASE = "http://localhost:11434"


class ModelSwapManager:
    """Singleton для управления загрузкой/выгрузкой моделей."""

    _instance: Optional["ModelSwapManager"] = None
    _lock: asyncio.Lock = None  # инициализируется при первом use

    def __init__(self):
        self._busy = False

    @classmethod
    def get_instance(cls) -> "ModelSwapManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Внутренние методы Ollama ──────────────────────────────────────────────

    async def _ollama_load(self, model: str, is_embed: bool = False) -> bool:
        """Загрузить модель в VRAM с keep_alive=-1."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if is_embed:
                    resp = await client.post(
                        f"{_OLLAMA_BASE}/api/embed",
                        json={"model": model, "input": "", "keep_alive": -1},
                    )
                else:
                    resp = await client.post(
                        f"{_OLLAMA_BASE}/api/generate",
                        json={"model": model, "prompt": "", "keep_alive": -1},
                    )
                if resp.status_code == 200:
                    logger.info(f"[SwapManager] ✅ {model} загружена в VRAM")
                    return True
                logger.warning(f"[SwapManager] ⚠️ {model} load failed: {resp.status_code}")
                return False
        except Exception as e:
            logger.warning(f"[SwapManager] ⚠️ load {model}: {e}")
            return False

    async def _ollama_unload(self, model: str, is_embed: bool = False) -> bool:
        """Выгрузить модель из VRAM (keep_alive=0)."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if is_embed:
                    resp = await client.post(
                        f"{_OLLAMA_BASE}/api/embed",
                        json={"model": model, "input": "", "keep_alive": 0},
                    )
                else:
                    resp = await client.post(
                        f"{_OLLAMA_BASE}/api/generate",
                        json={"model": model, "prompt": "", "keep_alive": 0},
                    )
                if resp.status_code == 200:
                    logger.info(f"[SwapManager] 🔄 {model} выгружена из VRAM")
                    return True
                return False
        except Exception as e:
            logger.warning(f"[SwapManager] ⚠️ unload {model}: {e}")
            return False

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    async def acquire_coordinator(
        self,
        coordinator_model: str,
        embedding_model: str = "",
    ) -> bool:
        """Освободить VRAM и загрузить coordinator.

        Выгружает embedding_model (если задан), загружает coordinator.
        Возвращает True если coordinator успешно загружен.
        """
        logger.info(f"[SwapManager] 🔁 acquire coordinator: {coordinator_model}")
        self._busy = True
        # Выгружаем embedding чтобы освободить VRAM
        if embedding_model:
            await self._ollama_unload(embedding_model, is_embed=True)
        # Загружаем coordinator
        ok = await self._ollama_load(coordinator_model, is_embed=False)
        return ok

    async def release_coordinator(
        self,
        coordinator_model: str,
        embedding_model: str = "",
    ) -> None:
        """Выгрузить coordinator, вернуть embedding в VRAM."""
        logger.info(f"[SwapManager] 🔁 release coordinator: {coordinator_model}")
        # Выгружаем coordinator
        await self._ollama_unload(coordinator_model, is_embed=False)
        # Загружаем embedding обратно
        if embedding_model:
            await self._ollama_load(embedding_model, is_embed=True)
        self._busy = False
        logger.info("[SwapManager] ✅ VRAM восстановлен — embedding обратно в памяти")

    @property
    def is_busy(self) -> bool:
        """True если выполняется swap операция."""
        return self._busy


def get_swap_manager() -> ModelSwapManager:
    """Получить singleton SwapManager."""
    return ModelSwapManager.get_instance()
