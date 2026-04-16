"""ProviderFactory — создаёт провайдеры моделей по конфигурации.

Абстракция над конкретными движками (Ollama, llama-cpp-python и т.д.).
Читает config.yaml через config_loader и создаёт нужный провайдер.

Use:
    from pure_intellect.engines.provider import get_coordinator, get_generator
    response = await get_coordinator().generate(messages)
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pure_intellect.engines.config_loader import ModelConfig, get_config

if TYPE_CHECKING:
    pass


# ════════════════════════════════════════════════════════
# Базовый интерфейс провайдера
# ════════════════════════════════════════════════════════

class ModelProvider(ABC):
    """Абстрактный провайдер модели."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Генерировать ответ по сообщениям."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Проверить доступность провайдера."""
        ...

    @abstractmethod
    def info(self) -> dict:
        """Информация о провайдере для статистики."""
        ...


# ════════════════════════════════════════════════════════
# Ollama Provider
# ════════════════════════════════════════════════════════

class OllamaModelProvider(ModelProvider):
    """Провайдер Ollama."""

    def __init__(self, cfg: ModelConfig):
        self._cfg = cfg
        self._active_model = cfg.model
        self._engine = None  # lazy init

    def _get_engine(self):
        if self._engine is None:
            from pure_intellect.engines.ollama import OllamaEngine
            self._engine = OllamaEngine()
        return self._engine

    async def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        engine = self._get_engine()
        temp = temperature if temperature is not None else self._cfg.temperature
        tokens = max_tokens if max_tokens is not None else self._cfg.max_tokens

        try:
            result = await engine.chat(
                messages=messages,
                model=self._active_model,
                temperature=temp,
                max_tokens=tokens,
            )
            return result.content
        except Exception as e:
            # Fallback на альтернативную модель
            if self._cfg.fallback_model and self._active_model != self._cfg.fallback_model:
                import warnings
                warnings.warn(
                    f"[OllamaProvider] {self._active_model} failed ({e}), "
                    f"falling back to {self._cfg.fallback_model}"
                )
                self._active_model = self._cfg.fallback_model
                result = await engine.chat(
                    messages=messages,
                    model=self._active_model,
                    temperature=temp,
                    max_tokens=tokens,
                )
                return result.content
            raise

    async def is_available(self) -> bool:
        try:
            engine = self._get_engine()
            models = await engine.list_models()
            return self._cfg.model in models or any(
                self._cfg.model.split(":")[0] in m for m in models
            )
        except Exception:
            return False

    async def restore_primary(self) -> bool:
        """Попытаться вернуться к основной модели после fallback."""
        if self._active_model != self._cfg.model:
            avail = await self.is_available()
            if avail:
                self._active_model = self._cfg.model
                return True
        return False

    def info(self) -> dict:
        return {
            "provider": "ollama",
            "configured_model": self._cfg.model,
            "active_model": self._active_model,
            "fallback_model": self._cfg.fallback_model,
            "using_fallback": self._active_model != self._cfg.model,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }


# ════════════════════════════════════════════════════════
# Hardware Detection
# ════════════════════════════════════════════════════════

def detect_free_vram_mb() -> int:
    """Определить свободную VRAM в MB. Возвращает 0 если GPU недоступен."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return int(lines[0].strip())
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except Exception:
        pass

    return 0


def detect_optimal_gpu_layers(model_size_gb: float = 4.0) -> int:
    """Вычислить оптимальное число GPU слоёв на основе свободной VRAM.

    Логика:
    - Если VRAM > model_size × 1.2 → все слои (-1)
    - Если VRAM > model_size × 0.5 → часть слоёв
    - Если VRAM < 1GB → CPU only (0)
    """
    free_mb = detect_free_vram_mb()
    cfg = get_config()
    reserve_mb = cfg.hardware.vram_reserve_mb
    usable_mb = max(0, free_mb - reserve_mb)
    model_mb = model_size_gb * 1024

    if usable_mb <= 0:
        return 0  # CPU only

    ratio = usable_mb / model_mb
    if ratio >= 1.2:
        return -1  # всё на GPU
    elif ratio >= 0.5:
        # Частичный offload: пропорционально числу слоёв
        # Типичная 7B модель: ~32 слоя
        estimated_layers = 32
        return max(1, int(estimated_layers * ratio * 0.8))
    else:
        return 0  # CPU only


# ════════════════════════════════════════════════════════
# ProviderFactory
# ════════════════════════════════════════════════════════

class ProviderFactory:
    """Фабрика провайдеров моделей.

    Создаёт провайдеры по конфигурации из config.yaml.
    Поддерживает lazy initialization и кеширование.
    """

    _instances: dict[str, ModelProvider] = {}

    @classmethod
    def create(cls, cfg: ModelConfig, role: str = "unknown") -> ModelProvider:
        """Создать провайдер для конфигурации модели."""
        provider_type = cfg.provider.lower()

        if provider_type in ("ollama", "auto"):
            return OllamaModelProvider(cfg)

        # Будущие провайдеры добавляются здесь:
        # elif provider_type == "llamacpp":
        #     return LlamaCppModelProvider(cfg)

        # Default: Ollama
        import warnings
        warnings.warn(f"[ProviderFactory] Unknown provider '{provider_type}', using Ollama")
        return OllamaModelProvider(cfg)

    @classmethod
    def get_coordinator(cls) -> ModelProvider:
        """Получить провайдер координатора (singleton)."""
        if "coordinator" not in cls._instances:
            cfg = get_config().coordinator
            cls._instances["coordinator"] = cls.create(cfg, role="coordinator")
        return cls._instances["coordinator"]

    @classmethod
    def get_generator(cls) -> ModelProvider:
        """Получить провайдер генератора (singleton)."""
        if "generator" not in cls._instances:
            cfg = get_config().generator
            cls._instances["generator"] = cls.create(cfg, role="generator")
        return cls._instances["generator"]

    @classmethod
    def reset(cls) -> None:
        """Сбросить все провайдеры (используется при reload_config)."""
        cls._instances.clear()

    @classmethod
    async def status(cls) -> dict:
        """Получить статус всех провайдеров."""
        cfg = get_config()
        coordinator = cls.get_coordinator()
        generator = cls.get_generator()

        coord_available = await coordinator.is_available()
        gen_available = await generator.is_available()
        free_vram = detect_free_vram_mb()

        return {
            "coordinator": {
                **coordinator.info(),
                "available": coord_available,
            },
            "generator": {
                **generator.info(),
                "available": gen_available,
            },
            "hardware": {
                "free_vram_mb": free_vram,
                "gpu_available": free_vram > 0,
                "auto_gpu_layers": cfg.hardware.auto_gpu_layers,
                "vram_overflow_strategy": cfg.hardware.vram_overflow_strategy,
            },
            "config_source": str(
                next(
                    (p for p in [
                        __import__("pathlib").Path.cwd() / "config.yaml"
                    ] if p.exists()),
                    "defaults"
                )
            ),
        }


# ── Convenience functions ────────────────────────────────

def get_coordinator() -> ModelProvider:
    """Получить провайдер координатора."""
    return ProviderFactory.get_coordinator()


def get_generator() -> ModelProvider:
    """Получить провайдер генератора."""
    return ProviderFactory.get_generator()


async def provider_status() -> dict:
    """Получить полный статус провайдеров."""
    return await ProviderFactory.status()
