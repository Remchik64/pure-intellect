"""Engines — провайдеры LLM моделей для Pure Intellect.

Иерархия:
    BaseEngine          ← абстрактный движок (base.py)
    OllamaEngine        ← Ollama HTTP клиент (ollama.py)

    ModelProvider       ← абстрактный провайдер (provider.py)
    OllamaModelProvider ← Ollama провайдер с fallback
    ProviderFactory     ← фабрика провайдеров из config.yaml

    ModelConfig         ← конфигурация модели (config_loader.py)
    AppConfig           ← полная конфигурация приложения
    get_config()        ← singleton конфигурации
    reload_config()     ← перечитать config.yaml

Быстрый старт:
    from pure_intellect.engines import get_coordinator, get_generator, get_config

    # Получить провайдер из config.yaml
    coordinator = get_coordinator()
    generator = get_generator()

    # Проверить конфигурацию
    cfg = get_config()
    print(cfg.coordinator.model)  # qwen2.5:3b
    print(cfg.generator.model)    # qwen2.5:7b
"""

from pure_intellect.engines.base import BaseEngine, LLMResponse
from pure_intellect.engines.ollama import OllamaEngine
from pure_intellect.engines.config_loader import (
    ModelConfig,
    HardwareConfig,
    MemoryConfig,
    AppConfig,
    get_config,
    reload_config,
    load_config,
)
from pure_intellect.engines.provider import (
    ModelProvider,
    OllamaModelProvider,
    ProviderFactory,
    get_coordinator,
    get_generator,
    provider_status,
    detect_free_vram_mb,
    detect_optimal_gpu_layers,
)

__all__ = [
    # Base
    "BaseEngine",
    "LLMResponse",
    # Ollama Engine
    "OllamaEngine",
    # Config
    "ModelConfig",
    "HardwareConfig",
    "MemoryConfig",
    "AppConfig",
    "get_config",
    "reload_config",
    "load_config",
    # Providers
    "ModelProvider",
    "OllamaModelProvider",
    "ProviderFactory",
    "get_coordinator",
    "get_generator",
    "provider_status",
    # Hardware
    "detect_free_vram_mb",
    "detect_optimal_gpu_layers",
]
