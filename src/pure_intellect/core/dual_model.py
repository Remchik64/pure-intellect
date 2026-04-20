"""DualModelRouter — маршрутизатор двух моделей Pure Intellect.

P6: Двойная дистилляция — разделение ролей между моделями.
Гибкость: Модели и параметры читаются из config.yaml.
           Можно менять модели без перезапуска сервера.

Архитектура:
  Coordinator (3B по умолчанию, быстрый, точный в структуре):
    - Создание координат сессии (_create_coordinate)
    - Классификация важности фактов (ImportanceTagger)
    - Определение намерения (Intent detection)
    → Малые, структурированные задачи: JSON output, classification, extraction

  Generator (7B по умолчанию, качественный, развёрнутый):
    - Основные ответы пользователю (run())
    - Сложные объяснения и рассуждения
    → Качественная генерация текста для конечного пользователя

Преимущества:
  - Coordinator = быстрый coordination overhead (~1-2s)
  - Generator = качественные ответы пользователю
  - Config-driven: меняй модели в config.yaml без перезапуска
  - CPU/GPU hybrid: auto-detect оптимальных GPU слоёв
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

# Дефолтные значения (используются если config.yaml не найден)
_DEFAULT_COORDINATOR = "qwen2.5:3b"
_DEFAULT_GENERATOR = "qwen2.5:7b"

# Алиасы для обратной совместимости
COORDINATOR_MODEL = _DEFAULT_COORDINATOR
GENERATOR_MODEL = _DEFAULT_GENERATOR

# Настройки Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATE_TIMEOUT = 120  # секунд для генератора
COORDINATE_TIMEOUT = 30  # секунд для координатора


def _load_models_from_config() -> tuple[str, str]:
    """Прочитать модели из config.yaml.

    Returns: (coordinator_model, generator_model)
    При ошибке возвращает дефолтные значения.
    """
    try:
        from pure_intellect.engines.config_loader import get_config
        cfg = get_config()
        return cfg.coordinator.model, cfg.generator.model
    except Exception:
        return _DEFAULT_COORDINATOR, _DEFAULT_GENERATOR


class DualModelRouter:
    """Маршрутизатор запросов между coordinator и generator моделями.

    Config-driven: модели читаются из config.yaml при инициализации.
    Параметры конструктора override значения из config (для тестов).

    Использование:
        router = DualModelRouter()                              # из config.yaml
        router = DualModelRouter(coordinator_model="qwen2.5:3b")  # override

        # Для структурированных задач (координаты, теги)
        result = router.coordinate(messages, temperature=0.1)

        # Для основной генерации (ответы пользователю)
        response = router.generate(messages, temperature=0.7)
    """

    def __init__(
        self,
        coordinator_model: Optional[str] = None,
        generator_model: Optional[str] = None,
        ollama_url: str = OLLAMA_BASE_URL,
    ):
        # Читаем из config.yaml если не переданы явно
        cfg_coordinator, cfg_generator = _load_models_from_config()
        self.coordinator_model = coordinator_model or cfg_coordinator
        self.generator_model = generator_model or cfg_generator
        self.ollama_url = ollama_url

        logger.info(
            f"[dual_model] Init: coordinator={self.coordinator_model!r}, "
            f"generator={self.generator_model!r}"
        )

        # Статистика
        self._coordinator_calls: int = 0
        self._generator_calls: int = 0
        self._coordinator_tokens: int = 0
        self._generator_tokens: int = 0

        # Доступность generator (7B может быть ещё не скачан)
        self._generator_available: bool | None = None

    def _check_generator_available(self) -> bool:
        """Проверить доступность generator модели (кешируем результат)."""
        if self._generator_available is not None:
            return self._generator_available

        try:
            payload = json.dumps({"name": self.generator_model}).encode()
            req = urllib.request.Request(
                f"{self.ollama_url}/api/show",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                self._generator_available = "modelfile" in data or "details" in data
                if self._generator_available:
                    logger.info(
                        f"[dual_model] Generator available: {self.generator_model}"
                    )
                else:
                    logger.warning(
                        f"[dual_model] Generator not ready: {self.generator_model}"
                    )
        except Exception as e:
            self._generator_available = False
            logger.warning(
                f"[dual_model] Generator unavailable: {self.generator_model} ({e})"
            )

        return self._generator_available

    def refresh_generator_check(self) -> bool:
        """Принудительно перепроверить доступность generator."""
        self._generator_available = None
        return self._check_generator_available()

    def reload_from_config(self) -> None:
        """Перечитать config.yaml и обновить модели без перезапуска."""
        try:
            from pure_intellect.engines.config_loader import reload_config
            reload_config()
            cfg_coordinator, cfg_generator = _load_models_from_config()
            old_coordinator = self.coordinator_model
            old_generator = self.generator_model
            self.coordinator_model = cfg_coordinator
            self.generator_model = cfg_generator
            self._generator_available = None  # сброс кеша доступности
            logger.info(
                f"[dual_model] Reloaded config: "
                f"coordinator {old_coordinator!r} → {self.coordinator_model!r}, "
                f"generator {old_generator!r} → {self.generator_model!r}"
            )
        except Exception as e:
            logger.error(f"[dual_model] Failed to reload config: {e}")

    def _call_ollama(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> tuple[str, int, int]:
        """Синхронный вызов Ollama API.

        Returns: (response_text, prompt_tokens, completion_tokens)
        """
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "options": {
                "num_ctx": 32768,
                "num_gpu": -1,     # GPU+CPU hybrid: load max layers in VRAM, rest on CPU
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return content, prompt_tokens, completion_tokens

    def coordinate(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Вызов coordinator модели для структурированных задач.

        Низкая температура для точного JSON/структурированного вывода.
        """
        try:
            content, pt, ct = self._call_ollama(
                messages=messages,
                model=self.coordinator_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=COORDINATE_TIMEOUT,
            )
            self._coordinator_calls += 1
            self._coordinator_tokens += pt + ct
            logger.debug(
                f"[coordinator:{self.coordinator_model}] "
                f"{pt}+{ct} tokens"
            )
            return content
        except Exception as e:
            logger.error(f"[coordinator] Call failed: {e}")
            return ""

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> tuple[str, int, int]:
        """Вызов generator модели (7B если доступна, иначе fallback).

        Returns: (response_text, prompt_tokens, completion_tokens)
        """
        # Выбираем модель
        if self._check_generator_available():
            model = self.generator_model
            timeout = GENERATE_TIMEOUT
            label = "generator"
        else:
            # Fallback на coordinator если generator недоступен
            model = self.coordinator_model
            timeout = GENERATE_TIMEOUT
            label = "coordinator-fallback"
            logger.debug(
                f"[dual_model] Generator unavailable, using coordinator as fallback"
            )

        try:
            content, pt, ct = self._call_ollama(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            self._generator_calls += 1
            self._generator_tokens += pt + ct
            logger.debug(
                f"[{label}:{model}] {pt}+{ct} tokens"
            )
            return content, pt, ct
        except Exception as e:
            logger.error(f"[generator] Call failed: {e}")
            return "", 0, 0

    def stats(self) -> dict:
        """Статистика вызовов dual model."""
        return {
            "coordinator_model": self.coordinator_model,
            "generator_model": self.generator_model,
            "generator_available": self._generator_available,
            "coordinator_calls": self._coordinator_calls,
            "generator_calls": self._generator_calls,
            "coordinator_tokens": self._coordinator_tokens,
            "generator_tokens": self._generator_tokens,
            "total_calls": self._coordinator_calls + self._generator_calls,
            "total_tokens": self._coordinator_tokens + self._generator_tokens,
        }

    def config_info(self) -> dict:
        """Информация о конфигурации из config.yaml."""
        try:
            from pure_intellect.engines.config_loader import get_config, _find_config_yaml
            cfg = get_config()
            config_path = _find_config_yaml()
            return {
                "config_file": str(config_path) if config_path else "defaults",
                "coordinator": {
                    "model": self.coordinator_model,
                    "temperature": cfg.coordinator.temperature,
                    "max_tokens": cfg.coordinator.max_tokens,
                    "gpu_layers": cfg.coordinator.gpu_layers,
                },
                "generator": {
                    "model": self.generator_model,
                    "temperature": cfg.generator.temperature,
                    "max_tokens": cfg.generator.max_tokens,
                    "gpu_layers": cfg.generator.gpu_layers,
                    "fallback_model": cfg.generator.fallback_model,
                },
                "hardware": {
                    "auto_gpu_layers": cfg.hardware.auto_gpu_layers,
                    "vram_reserve_mb": cfg.hardware.vram_reserve_mb,
                    "vram_overflow_strategy": cfg.hardware.vram_overflow_strategy,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def __repr__(self) -> str:
        gen_status = "✅" if self._generator_available else (
            "⏳" if self._generator_available is None else "❌"
        )
        return (
            f"DualModelRouter("
            f"coordinator={self.coordinator_model!r}, "
            f"generator={self.generator_model!r} {gen_status})"
        )
