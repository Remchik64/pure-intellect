"""Загрузчик конфигурации моделей из config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class ModelConfig:
    """Конфигурация одной модели."""
    provider: str = "ollama"          # ollama | llamacpp | sentence_transformers | auto
    model: str = "qwen2.5:3b"
    gpu_layers: str | int = "auto"    # auto | -1 | 0 | N
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    fallback_model: str | None = None
    device: str = "auto"              # для embeddings: auto | cuda | cpu
    extra: dict = field(default_factory=dict)

    def resolved_gpu_layers(self, available_vram_mb: int = 0) -> int:
        """Вернуть конкретное число GPU слоёв.

        auto → -1 если VRAM > 2GB, иначе 0 (CPU fallback)
        -1   → все слои на GPU
        0    → только CPU
        N    → ровно N слоёв на GPU
        """
        if isinstance(self.gpu_layers, int):
            return self.gpu_layers

        if self.gpu_layers == "auto":
            if available_vram_mb > 2048:
                return -1   # всё на GPU
            elif available_vram_mb > 0:
                return 20   # частичный offload
            else:
                return 0    # CPU only

        try:
            return int(self.gpu_layers)
        except (ValueError, TypeError):
            return -1


@dataclass
class HardwareConfig:
    """Аппаратные ресурсы."""
    auto_gpu_layers: bool = True
    vram_reserve_mb: int = 512
    cpu_threads: int = 0
    vram_overflow_strategy: str = "offload"  # offload | downgrade | fail


@dataclass
class MemoryConfig:
    """Параметры памяти."""
    context_window_messages: int = 12
    keep_after_reset: int = 6
    working_memory_tokens: int = 500
    max_storage_facts: int = 1000
    # R1: Мета-координата
    meta_coordinate_every: int = 4   # объединять каждые N координат в мета
    # R2/R3: RAM management
    max_hot_facts: int = 50           # максимум горячих фактов в WorkingMemory
    hot_evict_threshold: float = 0.2  # порог внимания для выгрузки из RAM


@dataclass
class AppConfig:
    """Полная конфигурация приложения."""
    coordinator: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="ollama",
        model="qwen2.5:3b",
        temperature=0.2,
        max_tokens=400,
        timeout=60,
    ))
    generator: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="ollama",
        model="qwen2.5:7b",
        temperature=0.7,
        max_tokens=2048,
        timeout=120,
        fallback_model="qwen2.5:3b",
    ))
    embedder: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="sentence_transformers",
        model="all-MiniLM-L6-v2",
        device="auto",
    ))
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def _find_config_yaml() -> Path | None:
    """Найти config.yaml в стандартных местах."""
    candidates: list[Path] = []

    # 1. Env variable (highest priority)
    env_val = os.environ.get("PURE_INTELLECT_CONFIG", "")
    if env_val:
        candidates.append(Path(env_val))

    # 2. Windows AppData
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        candidates.append(Path(appdata) / "PureIntellect" / "config.yaml")

    # 3. Linux/macOS XDG
    candidates.append(Path.home() / ".config" / "pure-intellect" / "config.yaml")

    # 4. Current working directory (may raise PermissionError)
    try:
        candidates.append(Path.cwd() / "config.yaml")
    except PermissionError:
        pass

    # 5. Project root (next to src/)
    candidates.append(Path(__file__).parent.parent.parent.parent / "config.yaml")

    # 6. System-wide
    candidates.append(Path("/etc/pure-intellect/config.yaml"))

    for path in candidates:
        try:
            if path.exists():
                return path
        except (PermissionError, OSError):
            continue
    return None


def _parse_model_config(data: dict) -> ModelConfig:
    """Создать ModelConfig из словаря YAML."""
    return ModelConfig(
        provider=data.get("provider", "ollama"),
        model=data.get("model", "qwen2.5:3b"),
        gpu_layers=data.get("gpu_layers", "auto"),
        temperature=float(data.get("temperature", 0.7)),
        max_tokens=int(data.get("max_tokens", 2048)),
        timeout=int(data.get("timeout", 120)),
        fallback_model=data.get("fallback_model"),
        device=data.get("device", "auto"),
        extra={k: v for k, v in data.items() if k not in {
            "provider", "model", "gpu_layers", "temperature",
            "max_tokens", "timeout", "fallback_model", "device"
        }},
    )


def load_config() -> AppConfig:
    """Загрузить конфигурацию из config.yaml или вернуть defaults."""
    if not HAS_YAML:
        return AppConfig()

    config_path = _find_config_yaml()
    if config_path is None:
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        models = raw.get("models", {})
        hardware_raw = raw.get("hardware", {})
        memory_raw = raw.get("memory", {})

        cfg = AppConfig()

        if "coordinator" in models:
            cfg.coordinator = _parse_model_config(models["coordinator"])
        if "generator" in models:
            cfg.generator = _parse_model_config(models["generator"])
        if "embedder" in models:
            cfg.embedder = _parse_model_config(models["embedder"])

        cfg.hardware = HardwareConfig(
            auto_gpu_layers=hardware_raw.get("auto_gpu_layers", True),
            vram_reserve_mb=int(hardware_raw.get("vram_reserve_mb", 512)),
            cpu_threads=int(hardware_raw.get("cpu_threads", 0)),
            vram_overflow_strategy=hardware_raw.get("vram_overflow_strategy", "offload"),
        )

        cfg.memory = MemoryConfig(
            context_window_messages=int(memory_raw.get("context_window_messages", 12)),
            keep_after_reset=int(memory_raw.get("keep_after_reset", 6)),
            working_memory_tokens=int(memory_raw.get("working_memory_tokens", 500)),
            max_storage_facts=int(memory_raw.get("max_storage_facts", 1000)),
        )

        return cfg

    except Exception as e:
        import warnings
        warnings.warn(f"[config_loader] Failed to load config.yaml: {e}. Using defaults.")
        return AppConfig()


# ── Singleton ────────────────────────────────────────────
_app_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Получить глобальный экземпляр конфигурации (singleton)."""
    global _app_config
    if _app_config is None:
        _app_config = load_config()
    return _app_config


def reload_config() -> AppConfig:
    """Перечитать config.yaml и обновить глобальный экземпляр."""
    global _app_config
    _app_config = load_config()
    return _app_config
