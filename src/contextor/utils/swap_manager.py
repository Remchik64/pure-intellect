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

from ..config import settings

logger = logging.getLogger(__name__)


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
                        f"{settings.ollama_url}/api/embed",
                        json={"model": model, "input": "", "keep_alive": -1},
                    )
                else:
                    resp = await client.post(
                        f"{settings.ollama_url}/api/generate",
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
                        f"{settings.ollama_url}/api/embed",
                        json={"model": model, "input": "", "keep_alive": 0},
                    )
                else:
                    resp = await client.post(
                        f"{settings.ollama_url}/api/generate",
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


    async def _wait_for_unload(self, model: str, timeout: float = 30.0) -> bool:
        """Poll /api/ps until model is gone from VRAM or timeout."""
        import httpx, asyncio
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{settings.ollama_url}/api/ps")
                    if resp.status_code == 200:
                        loaded = [m.get("name", "") for m in resp.json().get("models", [])]
                        # Check if model is still loaded (compare base name)
                        base = model.split(":")[0]
                        still_loaded = any(base in m for m in loaded)
                        if not still_loaded:
                            logger.info(f"[SwapManager] ✅ {model} подтверждено выгружена из VRAM")
                            return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        logger.warning(f"[SwapManager] ⚠️ Timeout ожидания выгрузки {model}")
        return False

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

    async def acquire_utility(
        self,
        utility_model: str,
        generator_model: str,
        coordinator_model: str = "",
        embedding_model: str = "",
    ) -> bool:
        """Освободить VRAM и загрузить utility эксклюзивно.

        Выгружает генератора, координатора и эмбеддер в RAM (keep_alive=0),
        чтобы освободить все 12 ГБ VRAM для обработки огромных текстов.
        """
        logger.info(f"[SwapManager] 🔁 acquire utility: {utility_model}")
        self._busy = True
        if generator_model:
            await self._ollama_unload(generator_model, is_embed=False)
            await self._wait_for_unload(generator_model)
        if coordinator_model:
            await self._ollama_unload(coordinator_model, is_embed=False)
            await self._wait_for_unload(coordinator_model)
        if embedding_model:
            await self._ollama_unload(embedding_model, is_embed=True)
            await self._wait_for_unload(embedding_model)
        # Extra safety: small delay for Windows VRAM release
        import asyncio as _asyncio
        await _asyncio.sleep(1.0)
        ok = await self._ollama_load(utility_model, is_embed=False)
        return ok

    async def release_utility(
        self,
        utility_model: str,
        generator_model: str,
        embedding_model: str = "",
    ) -> None:
        """Выгрузить utility, вернуть generator в VRAM."""
        logger.info(f"[SwapManager] 🔁 release utility: {utility_model}")
        await self._ollama_unload(utility_model, is_embed=False)
        if generator_model:
            await self._ollama_load(generator_model, is_embed=False)
        if embedding_model:
            await self._ollama_load(embedding_model, is_embed=True)
        self._busy = False
        logger.info("[SwapManager] ✅ VRAM восстановлен — generator и embedding обратно в памяти")

    @property
    def is_busy(self) -> bool:
        """True если выполняется swap операция."""
        return self._busy


def get_swap_manager() -> ModelSwapManager:
    """Получить singleton SwapManager."""
    return ModelSwapManager.get_instance()
