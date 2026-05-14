"""Engines — провайдеры LLM моделей для Contextor.

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
    from contextor.engines import get_coordinator, get_generator, get_config

    # Получить провайдер из config.yaml
    coordinator = get_coordinator()
    generator = get_generator()

    # Проверить конфигурацию
    cfg = get_config()
    print(cfg.coordinator.model)  # qwen3.5:2b
    print(cfg.generator.model)    # qwen3.5:9b
"""

from contextor.engines.base import BaseEngine, LLMResponse
from contextor.engines.ollama import OllamaEngine
from contextor.engines.config_loader import (
    ModelConfig,
    HardwareConfig,
    MemoryConfig,
    AppConfig,
    get_config,
    reload_config,
    load_config,
)
from contextor.engines.provider import (
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
